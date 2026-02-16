from pydantic import BaseModel


class COCOCategory(BaseModel):
    id: int
    name: str
    supercategory: str = ""


class COCOImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int


class COCOAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: float
    segmentation: list
    iscrowd: int = 0
