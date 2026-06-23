"""ASGI entry point for the backend."""

from app.core.application import create_app

app = create_app()
