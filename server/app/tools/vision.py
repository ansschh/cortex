"""Vision tools — capture snapshots and describe scenes via vision LLM."""

from __future__ import annotations

import base64
import logging
from typing import Any

from server.app.tools.base import BaseTool
from server.app.vision.camera import CameraManager
from shared.schemas.events import SensitivityLevel
from shared.schemas.tool_calls import ToolResult

logger = logging.getLogger(__name__)


class VisionSnapshotTool(BaseTool):
    name = "vision.snapshot"
    description = "Capture a single frame from the camera. Returns base64 JPEG."
    parameters_schema = {
        "type": "object",
        "properties": {
            "camera_id": {
                "type": "string",
                "description": "Optional camera ID. Defaults to the first available camera.",
            },
        },
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, camera_manager: CameraManager):
        self._camera = camera_manager

    async def execute(self, **kwargs: Any) -> ToolResult:
        camera_id = kwargs.get("camera_id")
        try:
            jpeg_bytes, used_camera = await self._camera.capture(camera_id)
            b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            return ToolResult(
                tool_name=self.name, success=True,
                result={"image_base64": b64, "format": "jpeg", "camera_id": used_camera},
            )
        except Exception as e:
            logger.error(f"Snapshot failed: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))


class VisionDescribeTool(BaseTool):
    name = "vision.describe"
    description = "Capture a frame and describe what the camera sees using a vision LLM. Use this when the user asks about anything visual — what they're wearing, holding, what's around them, reading text on objects, etc."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What to look for or describe",
                "default": "Describe what you see in this image.",
            },
            "camera_id": {
                "type": "string",
                "description": "Optional camera ID",
            },
        },
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, camera_manager: CameraManager, vision_llm=None):
        self._camera = camera_manager
        self._llm = vision_llm

    async def execute(self, **kwargs: Any) -> ToolResult:
        question = kwargs.get("question", "Describe what you see in this image.")
        camera_id = kwargs.get("camera_id")

        # Capture frame
        try:
            jpeg_bytes, used_camera = await self._camera.capture(camera_id)
            b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"Capture failed: {e}")

        if self._llm is None:
            return ToolResult(
                tool_name=self.name, success=False,
                error="No vision LLM configured.",
            )

        # Send to vision LLM
        from server.app.llm.base import LLMMessage

        messages = [
            LLMMessage(role="user", content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]),
        ]

        try:
            response = await self._llm.chat(messages, temperature=0.3)
            return ToolResult(
                tool_name=self.name, success=True,
                result={"description": response.text, "camera_id": used_camera},
                display_card={
                    "card_type": "VisionCard",
                    "title": "Camera View",
                    "body": response.text,
                },
            )
        except Exception as e:
            logger.error(f"Vision LLM error: {e}")
            return ToolResult(tool_name=self.name, success=False, error=str(e))
