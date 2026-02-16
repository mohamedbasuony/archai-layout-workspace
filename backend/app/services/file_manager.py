"""Temporary file and task state management."""

import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict

from app.config import settings

TASK_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".tasks")


@dataclass
class TaskState:
    task_id: str
    status: str = "pending"  # pending | processing | completed | error
    progress: int = 0
    total: int = 0
    current_image: str = ""
    message: str = ""
    created_at: float = field(default_factory=time.time)
    # Result data
    coco_json: dict | None = None
    stats: dict | None = None
    annotated_image_path: str | None = None
    # Batch-specific
    batch_coco_list: list = field(default_factory=list)
    gallery: list = field(default_factory=list)  # list of {filename, annotated_path}
    stats_per_image: list = field(default_factory=list)
    stats_summary: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


_task_store: Dict[str, TaskState] = {}
_lock = Lock()


def create_task() -> TaskState:
    """Create a new task with a unique ID and working directory."""
    task_id = uuid.uuid4().hex[:12]
    task_dir = os.path.join(TASK_BASE_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    task = TaskState(task_id=task_id)
    with _lock:
        _task_store[task_id] = task
    return task


def get_task(task_id: str) -> TaskState | None:
    with _lock:
        return _task_store.get(task_id)


def get_task_dir(task_id: str) -> str:
    path = os.path.join(TASK_BASE_DIR, task_id)
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_expired_tasks():
    """Remove tasks older than TTL."""
    cutoff = time.time() - settings.task_ttl_minutes * 60
    with _lock:
        expired = [tid for tid, t in _task_store.items() if t.created_at < cutoff]
        for tid in expired:
            del _task_store[tid]
            task_dir = os.path.join(TASK_BASE_DIR, tid)
            shutil.rmtree(task_dir, ignore_errors=True)
