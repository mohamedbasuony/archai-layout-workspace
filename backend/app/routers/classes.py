from fastapi import APIRouter

from app.core.constants import CLASS_COLOR_MAP, coco_class_mapping

router = APIRouter(tags=["classes"])


@router.get("/classes")
def get_classes():
    """Return all 25 manuscript element classes with IDs and colors."""
    return {
        "classes": [
            {"id": cid, "name": name, "color": CLASS_COLOR_MAP.get(name, "#888888")}
            for name, cid in coco_class_mapping.items()
        ]
    }
