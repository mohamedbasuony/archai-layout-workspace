"""
Prediction endpoints — single image and batch processing.
"""

import asyncio
import json
import os
import tempfile
import zipfile
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from app.core.constants import FINAL_CLASSES
from app.core.model_runner import combine_and_filter_predictions
from app.services.coco_utils import filter_coco_by_classes, merge_coco_list, stats_from_coco
from app.services.file_manager import TaskState, create_task, get_task, get_task_dir
from app.services.inference import run_models_parallel
from app.services.visualization import draw_coco_on_image

router = APIRouter(tags=["predict"])

# Valid image extensions for batch processing
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
SKIP_PATTERNS = ["._", ".DS_Store", "Thumbs.db", "desktop.ini", "~$"]


def _secure_filename(name: str) -> str:
    # Prevent path traversal and keep filenames sane
    name = os.path.basename(name or "")
    return name if name else "upload.jpg"


def _parse_classes(classes: Optional[str]) -> list[str]:
    if not classes:
        return FINAL_CLASSES
    try:
        selected = json.loads(classes)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid 'classes' JSON.")

    if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected):
        raise HTTPException(status_code=422, detail="'classes' must be a JSON array of strings.")
    if not selected:
        raise HTTPException(status_code=422, detail="Please select at least one class.")
    return selected


def _safe_extract_zip(zip_path: str, extract_dir: str) -> None:
    """
    Extract zip safely (prevents Zip Slip path traversal).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_name = member.filename

            # Skip obvious junk folders/files
            base = os.path.basename(member_name)
            if not base:
                continue
            if any(base.startswith(p) for p in SKIP_PATTERNS) or base.startswith("."):
                continue

            dest_path = os.path.normpath(os.path.join(extract_dir, member_name))
            if not dest_path.startswith(os.path.abspath(extract_dir) + os.sep):
                # Path traversal attempt
                continue

            if member.is_dir():
                os.makedirs(dest_path, exist_ok=True)
                continue

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
                dst.write(src.read())


@router.post("/predict/single")
async def predict_single(
    image: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.3),
    classes: Optional[str] = Form(None),
):
    """Run 3 YOLO models on a single image and return combined COCO JSON."""
    selected_classes = _parse_classes(classes)

    # Create task and save uploaded image
    task = create_task()
    task_dir = get_task_dir(task.task_id)

    safe_name = _secure_filename(image.filename or "upload.jpg")
    img_path = os.path.join(task_dir, safe_name)

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=422, detail="Empty upload.")

    with open(img_path, "wb") as f:
        f.write(contents)

    # Quick image validation
    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            im.load()
            if im.size[0] <= 0 or im.size[1] <= 0:
                raise ValueError("Invalid image dimensions.")
    except Exception:
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid image.")

    # Run models (offload CPU-bound work)
    with tempfile.TemporaryDirectory() as tmp_dir:
        labels_folders = await asyncio.to_thread(
            run_models_parallel, img_path, tmp_dir, conf=confidence, iou=iou
        )
        coco_json = await asyncio.to_thread(combine_and_filter_predictions, img_path, labels_folders)

    # Filter and annotate
    filtered_coco = filter_coco_by_classes(coco_json, selected_classes)
    annotated_path = os.path.join(task_dir, "annotated.jpg")
    await asyncio.to_thread(
        draw_coco_on_image, img_path, filtered_coco, selected_classes, output_path=annotated_path
    )

    stats = stats_from_coco(filtered_coco)

    # Store results
    task.status = "completed"
    task.coco_json = filtered_coco
    task.stats = stats
    task.annotated_image_path = annotated_path

    return {
        "task_id": task.task_id,
        "coco_json": filtered_coco,
        "stats": stats,
        "annotated_image_url": f"/api/download/{task.task_id}/annotated_image",
    }


def _process_batch(task: TaskState, zip_path: str, conf: float, iou: float, selected_classes: list[str]):
    """Background worker for batch processing."""
    task_dir = get_task_dir(task.task_id)

    with tempfile.TemporaryDirectory() as extract_dir:
        _safe_extract_zip(zip_path, extract_dir)

        # Find valid images
        image_paths = []
        for root, dirs, files in os.walk(extract_dir):
            if "__MACOSX" in root:
                continue
            dirs[:] = [d for d in dirs if d != "__MACOSX" and not d.startswith(".")]

            for fn in files:
                if any(fn.startswith(p) for p in SKIP_PATTERNS) or fn.startswith("."):
                    continue
                if os.path.splitext(fn)[1].lower() not in IMAGE_EXTENSIONS:
                    continue

                full_path = os.path.join(root, fn)
                if not os.path.isfile(full_path) or os.path.getsize(full_path) == 0:
                    continue

                try:
                    with Image.open(full_path) as test_img:
                        test_img.verify()
                    with Image.open(full_path) as test_img:
                        test_img.load()
                        if test_img.size[0] == 0 or test_img.size[1] == 0:
                            continue
                    image_paths.append(full_path)
                except Exception:
                    continue

        task.total = len(image_paths)
        task.status = "processing"

        for idx, path in enumerate(sorted(image_paths)):
            fn = os.path.basename(path)
            task.current_image = fn
            task.progress = idx

            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    labels_folders = run_models_parallel(path, tmp_dir, conf=conf, iou=iou)
                    coco_json = combine_and_filter_predictions(path, labels_folders)

                filtered = filter_coco_by_classes(coco_json, selected_classes)
                task.batch_coco_list.append(filtered)

                ann_path = os.path.join(task_dir, f"annotated_{idx}.jpg")
                draw_coco_on_image(path, filtered, selected_classes, output_path=ann_path)
                task.gallery.append({"filename": fn, "annotated_path": ann_path})

                stats = stats_from_coco(filtered)
                task.stats_per_image.append({"image": fn, "stats": stats})
                for k, v in stats.items():
                    task.stats_summary[k] = task.stats_summary.get(k, 0) + v

            except Exception as e:
                task.errors.append(f"Error processing {fn}: {str(e)}")

        task.progress = task.total
        task.status = "completed"

        # Save merged COCO JSON
        if task.batch_coco_list:
            merged = merge_coco_list(task.batch_coco_list)
            task.coco_json = merged
            coco_path = os.path.join(task_dir, "annotations.json")
            with open(coco_path, "w") as f:
                json.dump(merged, f, indent=2)


@router.post("/predict/batch", status_code=202)
async def predict_batch(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.3),
    classes: Optional[str] = Form(None),
):
    """Start batch processing of a ZIP archive."""
    selected_classes = _parse_classes(classes)

    task = create_task()
    task_dir = get_task_dir(task.task_id)

    zip_path = os.path.join(task_dir, "upload.zip")
    contents = await zip_file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="Empty ZIP upload.")

    with open(zip_path, "wb") as f:
        f.write(contents)

    background_tasks.add_task(_process_batch, task, zip_path, confidence, iou, selected_classes)

    return {
        "task_id": task.task_id,
        "sse_url": f"/api/predict/batch/{task.task_id}/progress",
    }


@router.get("/predict/batch/{task_id}/progress")
async def batch_progress(task_id: str):
    """SSE endpoint streaming batch processing progress."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    async def event_stream():
        last_progress = -1
        while True:
            if task.progress != last_progress or task.status in ("completed", "error"):
                data = json.dumps(
                    {
                        "status": task.status,
                        "progress": task.progress,
                        "total": task.total,
                        "current_image": task.current_image,
                        "message": task.message,
                    }
                )
                yield f"data: {data}\n\n"
                last_progress = task.progress

            if task.status in ("completed", "error"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/predict/batch/{task_id}/results")
async def batch_results(task_id: str):
    """Get final results for a completed batch task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task.status != "completed":
        raise HTTPException(status_code=409, detail=f"Task is still {task.status}.")

    gallery = [
        {
            "filename": g["filename"],
            "annotated_url": f"/api/download/{task_id}/annotated/{idx}",
        }
        for idx, g in enumerate(task.gallery)
    ]

    return {
        "status": task.status,
        "total_processed": task.total - len(task.errors),
        "errors": task.errors,
        "coco_json": task.coco_json,
        "stats_per_image": task.stats_per_image,
        "stats_summary": task.stats_summary,
        "gallery": gallery,
    }
