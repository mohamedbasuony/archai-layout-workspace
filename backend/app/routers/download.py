"""File download endpoints."""

import io
import json
import os
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.services.file_manager import get_task, get_task_dir

router = APIRouter(tags=["download"])


@router.get("/download/{task_id}/coco_json")
async def download_coco_json(task_id: str):
    """Download COCO JSON annotations for a task."""
    task = get_task(task_id)
    if task is None or task.coco_json is None:
        raise HTTPException(status_code=404, detail="Task not found or no results available.")

    content = json.dumps(task.coco_json, indent=2)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="annotations_{task_id}.json"'},
    )


@router.get("/download/{task_id}/annotated_image")
async def download_annotated_image(task_id: str):
    """Download the annotated image for a single-image task."""
    task = get_task(task_id)
    if task is None or task.annotated_image_path is None:
        raise HTTPException(status_code=404, detail="Task not found or no annotated image.")

    if not os.path.exists(task.annotated_image_path):
        raise HTTPException(status_code=404, detail="Annotated image file not found.")

    return FileResponse(
        task.annotated_image_path,
        media_type="image/jpeg",
        filename=f"annotated_{task_id}.jpg",
    )


@router.get("/download/{task_id}/annotated/{index}")
async def download_batch_annotated(task_id: str, index: int):
    """Download a specific annotated image from a batch task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    if index < 0 or index >= len(task.gallery):
        raise HTTPException(status_code=404, detail="Image index out of range.")

    path = task.gallery[index]["annotated_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Annotated image file not found.")

    filename = task.gallery[index]["filename"]
    return FileResponse(path, media_type="image/jpeg", filename=f"annotated_{filename}")


@router.get("/download/{task_id}/results_zip")
async def download_results_zip(task_id: str):
    """Download a ZIP containing all annotated images and the merged COCO JSON."""
    task = get_task(task_id)
    if task is None or not task.gallery:
        raise HTTPException(status_code=404, detail="Task not found or no results.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add annotated images
        for g in task.gallery:
            if os.path.exists(g["annotated_path"]):
                zf.write(g["annotated_path"], f"images/{g['filename']}")

        # Add merged annotations
        if task.coco_json:
            zf.writestr("annotations.json", json.dumps(task.coco_json, indent=2))

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="results_{task_id}.zip"'},
    )

