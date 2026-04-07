"""ComfyUI-MotionCapture: Motion capture from video for ComfyUI."""

import os
import sys
import logging
from pathlib import Path

log = logging.getLogger("motioncapture")

log.info("loading...")
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
log.info(f"loaded {len(NODE_CLASS_MAPPINGS)} nodes directly")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
