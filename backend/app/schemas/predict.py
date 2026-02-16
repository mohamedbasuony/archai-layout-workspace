from pydantic import BaseModel


class PredictSingleResponse(BaseModel):
    task_id: str
    coco_json: dict
    stats: dict[str, int]
    annotated_image_url: str


class BatchInitResponse(BaseModel):
    task_id: str
    sse_url: str


class BatchProgress(BaseModel):
    status: str  # processing | completed | error
    progress: int
    total: int
    current_image: str = ""
    message: str = ""


class BatchResultImage(BaseModel):
    filename: str
    annotated_url: str


class BatchResults(BaseModel):
    status: str
    total_processed: int
    errors: list[str]
    coco_json: dict
    stats_per_image: list[dict]
    stats_summary: dict[str, int]
    gallery: list[BatchResultImage]
