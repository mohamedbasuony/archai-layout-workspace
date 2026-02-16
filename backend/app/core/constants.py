"""Constants shared across the application."""

# Maps catmus/zone model class names to unified COCO class names
catmus_zones_mapping = {
    "DefaultLine": "Main script black",
    "InterlinearLine": "Gloss",
    "MainZone": "Column",
    "DropCapitalZone": "Plain initial- coloured",
    "StampZone": "Illustrations",
    "GraphicZone": "Illustrations",
    "MarginTextZone": "Gloss",
    "MusicZone": "Music",
    "NumberingZone": "Page Number",
    "QuireMarksZone": "Quire Mark",
    "RunningTitleZone": "Running header",
    "TitlePageZone": "Column",
}

# 25 final classes with COCO category IDs
coco_class_mapping = {
    "Border": 1,
    "Table": 2,
    "Diagram": 3,
    "Main script black": 4,
    "Main script coloured": 5,
    "Variant script black": 6,
    "Variant script coloured": 7,
    "Historiated": 8,
    "Inhabited": 9,
    "Zoo - Anthropomorphic": 10,
    "Embellished": 11,
    "Plain initial- coloured": 12,
    "Plain initial - Highlighted": 13,
    "Plain initial - Black": 14,
    "Page Number": 15,
    "Quire Mark": 16,
    "Running header": 17,
    "Catchword": 18,
    "Gloss": 19,
    "Illustrations": 20,
    "Column": 21,
    "GraphicZone": 22,
    "MusicLine": 23,
    "MusicZone": 24,
    "Music": 25,
}

FINAL_CLASSES = list(coco_class_mapping.keys())

# Per-model YOLO class filters
MODEL_CLASSES = {
    "emanuskript": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20],
    "catmus": [1, 7],
    "zone": None,  # all classes
}

# Hex colors for annotation rendering
CLASS_COLOR_MAP = {
    "Border": "#8B0000",
    "Table": "#006400",
    "Diagram": "#00008B",
    "Main script black": "#FF0000",
    "Main script coloured": "#FF4500",
    "Variant script black": "#8B008B",
    "Variant script coloured": "#FF1493",
    "Historiated": "#FFD700",
    "Inhabited": "#FF8C00",
    "Zoo - Anthropomorphic": "#32CD32",
    "Embellished": "#FF00FF",
    "Plain initial- coloured": "#00CED1",
    "Plain initial - Highlighted": "#00BFFF",
    "Plain initial - Black": "#000000",
    "Page Number": "#DC143C",
    "Quire Mark": "#9932CC",
    "Running header": "#228B22",
    "Catchword": "#B22222",
    "Gloss": "#4169E1",
    "Illustrations": "#FF6347",
    "Column": "#2E8B57",
    "GraphicZone": "#8A2BE2",
    "MusicLine": "#20B2AA",
    "MusicZone": "#4682B4",
    "Music": "#1E90FF",
}
