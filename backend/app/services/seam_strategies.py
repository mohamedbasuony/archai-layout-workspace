"""Seam-aware retry strategies for tile-based OCR.

When OCR attempt 0 produces identical text to a retry, the tiling grid must
physically change so that seam lines shift.  This module provides three
concrete strategies:

  Strategy A  – grid_shift    : divide the image into a different MxN grid
  Strategy B  – seam_band_crop: remove narrow bands at previous seam locations
  Strategy C  – expand_overlap: expand each tile rect by 22 % into neighbours

Each strategy takes the *previous* tile boxes (pixel coords) and the image
dimensions, and returns **new** tile boxes that are guaranteed to differ.

A NO-OP guard ensures that if a strategy fails to produce different boxes,
we fall through to the next one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────
# Data types
# ───────────────────────────────────────────────────────────────────────

Box = tuple[int, int, int, int]          # (x1, y1, x2, y2)


@dataclass
class TilingPlan:
    """Full description of how tiles were generated for one OCR attempt."""
    strategy: str                        # "default" | "grid_shift" | "expand_overlap" | "seam_band_crop"
    grid: str                            # e.g. "1x3" (rows×cols)
    overlap_pct: float                   # effective overlap fraction
    tile_boxes: list[Box]                # pixel-coord boxes
    preproc: dict[str, Any] = field(default_factory=dict)   # any pre-processing params
    text_sha256: str = ""                # filled after OCR text is known

    def boxes_signature(self) -> str:
        """Deterministic hash of tile_boxes for NO-OP detection."""
        canon = json.dumps(sorted(self.tile_boxes), separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def _clip(x1: int, y1: int, x2: int, y2: int,
          img_w: int, img_h: int) -> Box:
    """Clip a box to image bounds."""
    return (
        max(0, min(x1, img_w - 1)),
        max(0, min(y1, img_h - 1)),
        max(1, min(x2, img_w)),
        max(1, min(y2, img_h)),
    )


def _box_area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _boxes_identical(a: list[Box], b: list[Box]) -> bool:
    """True when two box lists produce the same set of pixel rectangles."""
    return sorted(a) == sorted(b)


def _infer_grid(boxes: list[Box]) -> str:
    """Guess approximate grid shape (rows×cols) from box centres."""
    if not boxes:
        return "0x0"
    ys = sorted(set((b[1] + b[3]) // 2 for b in boxes))
    xs = sorted(set((b[0] + b[2]) // 2 for b in boxes))

    # Cluster ys
    rows = 1
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] > 50:
            rows += 1
    # Cluster xs
    cols = 1
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] > 50:
            cols += 1
    return f"{rows}x{cols}"


def _seam_y_coords(boxes: list[Box]) -> list[int]:
    """Identify horizontal seam Y-coordinates between vertically adjacent tiles."""
    if len(boxes) < 2:
        return []
    # Sort by vertical centre
    sorted_boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    seams: list[int] = []
    for i in range(len(sorted_boxes) - 1):
        b_top = sorted_boxes[i]
        b_bot = sorted_boxes[i + 1]
        seam_y = (b_top[3] + b_bot[1]) // 2
        seams.append(seam_y)
    return seams


def _seam_x_coords(boxes: list[Box]) -> list[int]:
    """Identify vertical seam X-coordinates between horizontally adjacent tiles."""
    if len(boxes) < 2:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)
    seams: list[int] = []
    for i in range(len(sorted_boxes) - 1):
        b_left = sorted_boxes[i]
        b_right = sorted_boxes[i + 1]
        seam_x = (b_left[2] + b_right[0]) // 2
        seams.append(seam_x)
    return seams


# ───────────────────────────────────────────────────────────────────────
# Strategy A — grid_shift
# ───────────────────────────────────────────────────────────────────────

_GRID_SEQUENCE: list[tuple[int, int]] = [
    # (rows, cols) — deliberately different from defaults
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
    (4, 2),
]


def grid_shift(
    prev_boxes: list[Box],
    img_w: int,
    img_h: int,
    *,
    attempt_idx: int = 1,
    overlap_pct: float = 0.15,
) -> TilingPlan:
    """Divide the image into a new grid that differs from *prev_boxes*.

    Cycles through ``_GRID_SEQUENCE`` picking the first grid whose box
    signature differs from *prev_boxes*.
    """
    prev_sig = TilingPlan(
        strategy="prev", grid="", overlap_pct=0, tile_boxes=prev_boxes,
    ).boxes_signature()

    for rows, cols in _GRID_SEQUENCE:
        boxes = _make_grid(rows, cols, img_w, img_h, overlap_pct)
        plan = TilingPlan(
            strategy="grid_shift",
            grid=f"{rows}x{cols}",
            overlap_pct=overlap_pct,
            tile_boxes=boxes,
            preproc={"source": "grid_shift", "attempt_idx": attempt_idx},
        )
        if plan.boxes_signature() != prev_sig:
            logger.info(
                "grid_shift: switched to %dx%d grid (%d tiles) for attempt %d",
                rows, cols, len(boxes), attempt_idx,
            )
            return plan

    # Fallback: offset grid by 1/3 tile height
    rows, cols = 3, 1
    boxes = _make_grid(rows, cols, img_w, img_h, overlap_pct, y_offset_frac=0.33)
    return TilingPlan(
        strategy="grid_shift",
        grid=f"{rows}x{cols}_offset",
        overlap_pct=overlap_pct,
        tile_boxes=boxes,
        preproc={"source": "grid_shift_fallback", "attempt_idx": attempt_idx},
    )


def _make_grid(
    rows: int,
    cols: int,
    img_w: int,
    img_h: int,
    overlap_pct: float,
    y_offset_frac: float = 0.0,
) -> list[Box]:
    """Create a rows×cols grid of rectangular tiles with *overlap_pct* overlap."""
    if rows <= 0 or cols <= 0:
        return [_clip(0, 0, img_w, img_h, img_w, img_h)]

    tile_h = max(1, img_h // rows)
    tile_w = max(1, img_w // cols)
    ov_y = max(4, int(tile_h * overlap_pct))
    ov_x = max(4, int(tile_w * overlap_pct))
    y_off = int(tile_h * y_offset_frac)

    boxes: list[Box] = []
    for r in range(rows):
        for c in range(cols):
            y1 = r * tile_h + y_off
            y2 = y1 + tile_h
            x1 = c * tile_w
            x2 = x1 + tile_w

            # add overlap
            if r > 0:
                y1 = max(0, y1 - ov_y)
            if r < rows - 1:
                y2 = min(img_h, y2 + ov_y)
            if c > 0:
                x1 = max(0, x1 - ov_x)
            if c < cols - 1:
                x2 = min(img_w, x2 + ov_x)

            boxes.append(_clip(x1, y1, x2, y2, img_w, img_h))
    return boxes


# ───────────────────────────────────────────────────────────────────────
# Strategy B — seam_band_crop
# ───────────────────────────────────────────────────────────────────────

SEAM_BAND_PX: int = 16   # strip ±16 px around each horizontal seam


def seam_band_crop(
    prev_boxes: list[Box],
    img_w: int,
    img_h: int,
    *,
    band_px: int = SEAM_BAND_PX,
    attempt_idx: int = 1,
) -> TilingPlan:
    """Remove narrow bands at previous seam locations, producing new tiles
    that avoid the seam regions entirely."""
    seam_ys = _seam_y_coords(prev_boxes)
    seam_xs = _seam_x_coords(prev_boxes)

    if not seam_ys and not seam_xs:
        # No seams detected → fall through to grid_shift
        return grid_shift(prev_boxes, img_w, img_h, attempt_idx=attempt_idx)

    # Build new tile boxes by splitting between seam exclusion zones
    y_boundaries = sorted(set([0] + [max(0, sy - band_px) for sy in seam_ys]
                               + [min(img_h, sy + band_px) for sy in seam_ys]
                               + [img_h]))

    # Filter out tiny slivers
    y_edges: list[int] = [y_boundaries[0]]
    for y in y_boundaries[1:]:
        if y - y_edges[-1] >= 40:   # minimum band height
            y_edges.append(y)
    if y_edges[-1] != img_h:
        y_edges.append(img_h)

    # Similarly for x seams
    x_boundaries = sorted(set([0] + [max(0, sx - band_px) for sx in seam_xs]
                               + [min(img_w, sx + band_px) for sx in seam_xs]
                               + [img_w]))
    x_edges: list[int] = [x_boundaries[0]]
    for x in x_boundaries[1:]:
        if x - x_edges[-1] >= 40:
            x_edges.append(x)
    if x_edges[-1] != img_w:
        x_edges.append(img_w)

    boxes: list[Box] = []
    for i in range(len(y_edges) - 1):
        for j in range(len(x_edges) - 1):
            box = _clip(x_edges[j], y_edges[i], x_edges[j + 1], y_edges[i + 1],
                        img_w, img_h)
            if _box_area(box) >= 40 * 40:     # skip tiny slivers
                boxes.append(box)

    # Exclude bands that overlap with seam zones
    filtered: list[Box] = []
    for box in boxes:
        bx1, by1, bx2, by2 = box
        in_seam = False
        for sy in seam_ys:
            if by1 <= sy + band_px and by2 >= sy - band_px and (by2 - by1) < 3 * band_px:
                in_seam = True
                break
        if not in_seam:
            filtered.append(box)

    if not filtered:
        filtered = boxes   # keep all if filtering removed everything

    nrows = len(set((b[1] + b[3]) // 2 for b in filtered))
    ncols = len(set((b[0] + b[2]) // 2 for b in filtered))

    return TilingPlan(
        strategy="seam_band_crop",
        grid=f"{nrows}x{ncols}",
        overlap_pct=0.0,
        tile_boxes=filtered,
        preproc={
            "seam_ys": seam_ys,
            "seam_xs": seam_xs,
            "band_px": band_px,
            "attempt_idx": attempt_idx,
        },
    )


# ───────────────────────────────────────────────────────────────────────
# Strategy C — expand_overlap
# ───────────────────────────────────────────────────────────────────────

def expand_overlap(
    prev_boxes: list[Box],
    img_w: int,
    img_h: int,
    *,
    expand_pct: float = 0.22,
    attempt_idx: int = 1,
) -> TilingPlan:
    """Expand each previous tile by *expand_pct* into its neighbours.

    This changes the crop region for every tile, so seams get different
    context on each side.
    """
    boxes: list[Box] = []
    for x1, y1, x2, y2 in prev_boxes:
        w = x2 - x1
        h = y2 - y1
        dx = max(4, int(w * expand_pct))
        dy = max(4, int(h * expand_pct))
        boxes.append(_clip(x1 - dx, y1 - dy, x2 + dx, y2 + dy, img_w, img_h))

    return TilingPlan(
        strategy="expand_overlap",
        grid=_infer_grid(boxes),
        overlap_pct=expand_pct,
        tile_boxes=boxes,
        preproc={"expand_pct": expand_pct, "attempt_idx": attempt_idx},
    )


# ───────────────────────────────────────────────────────────────────────
# NO-OP guard + strategy selection
# ───────────────────────────────────────────────────────────────────────

_STRATEGY_CHAIN = [
    grid_shift,
    seam_band_crop,
    expand_overlap,
]


def select_retry_strategy(
    prev_plan: TilingPlan,
    img_w: int,
    img_h: int,
    *,
    prev_text_sha256: str = "",
    attempt_idx: int = 1,
) -> TilingPlan:
    """Pick a retry strategy that is guaranteed to change tile_boxes.

    Walks the strategy chain; the first strategy whose box signature
    differs from *prev_plan* wins.  If none do, returns the last one
    (the expand_overlap, which always changes geometry unless image is 1×1).
    """
    prev_sig = prev_plan.boxes_signature()

    for strategy_fn in _STRATEGY_CHAIN:
        plan = strategy_fn(
            prev_plan.tile_boxes, img_w, img_h, attempt_idx=attempt_idx,
        )
        if plan.boxes_signature() != prev_sig:
            logger.info(
                "select_retry_strategy: chose %s (sig %s → %s)",
                plan.strategy, prev_sig[:8], plan.boxes_signature()[:8],
            )
            return plan
        logger.info(
            "select_retry_strategy: %s produced same boxes; trying next",
            plan.strategy,
        )

    # Final fallback — guaranteed different: shift grid by 1/4 of image
    quarter_h = max(1, img_h // 4)
    boxes = [
        _clip(0, 0, img_w, img_h // 2 + quarter_h, img_w, img_h),
        _clip(0, img_h // 4, img_w, 3 * img_h // 4 + quarter_h, img_w, img_h),
        _clip(0, img_h // 2, img_w, img_h, img_w, img_h),
    ]
    return TilingPlan(
        strategy="fallback_quarter_shift",
        grid="3x1_shift25",
        overlap_pct=0.25,
        tile_boxes=boxes,
        preproc={"attempt_idx": attempt_idx, "reason": "all_strategies_no-op"},
    )


def is_noop_retry(
    prev_text_sha256: str,
    new_text_sha256: str,
    prev_plan: TilingPlan,
    new_plan: TilingPlan,
) -> bool:
    """Return True if a retry produced no meaningful change.

    Checks both text identity and tile-box identity.
    """
    if new_text_sha256 and prev_text_sha256 and new_text_sha256 == prev_text_sha256:
        return True
    if _boxes_identical(prev_plan.tile_boxes, new_plan.tile_boxes):
        return True
    return False


# ───────────────────────────────────────────────────────────────────────
# Convert TilingPlan → SaiaOCRLocationSuggestion-like dicts
# ───────────────────────────────────────────────────────────────────────

def plan_to_suggestions(
    plan: TilingPlan,
    img_w: int,
    img_h: int,
) -> list[dict[str, Any]]:
    """Convert tile boxes into location-suggestion dicts the OCR agent understands.

    Returns dicts with ``region_id``, ``category``, ``bbox_xywh`` keys.
    """
    suggestions: list[dict[str, Any]] = []
    for idx, (x1, y1, x2, y2) in enumerate(plan.tile_boxes):
        w = x2 - x1
        h = y2 - y1
        suggestions.append({
            "region_id": f"seam_retry_{plan.strategy}_{idx}",
            "category": "text_region",
            "bbox_xywh": [float(x1), float(y1), float(w), float(h)],
        })
    return suggestions


def default_plan_from_suggestions(
    suggestions: Sequence[dict[str, Any]],
    img_w: int,
    img_h: int,
) -> TilingPlan:
    """Build a TilingPlan from the original location_suggestions.

    Used to capture attempt 0's tiling geometry for the NO-OP guard.
    """
    boxes: list[Box] = []
    for s in suggestions:
        bbox = s.get("bbox_xywh") or getattr(s, "bbox_xywh", None) or []
        if len(bbox) >= 4:
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            boxes.append(_clip(x1, y1, x2, y2, img_w, img_h))
    if not boxes:
        # Fallback: full image as single tile
        boxes = [_clip(0, 0, img_w, img_h, img_w, img_h)]

    return TilingPlan(
        strategy="default",
        grid=_infer_grid(boxes),
        overlap_pct=0.12,      # nominal default
        tile_boxes=boxes,
    )
