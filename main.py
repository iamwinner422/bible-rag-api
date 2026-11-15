"""Compatibility entry point for deployments expecting ``main:app``."""

from app.main import app

__all__ = ["app"]
