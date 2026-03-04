"""Local vision LLM provider — SmolVLM-500M on GPU for real-time scene description.

Zero API cost, ~1s inference, 0.97 GB VRAM. Used by the always-on vision context
to continuously describe what the camera sees.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any, AsyncIterator, Optional

import torch
from PIL import Image

from server.app.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class LocalVisionProvider(LLMProvider):
    """SmolVLM-500M for fast local scene description."""

    def __init__(self, device: str = "cuda"):
        # Auto-detect CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available for SmolVLM — falling back to CPU")
            device = "cpu"
        self._device = device
        self._model = None
        self._processor = None
        self._loaded = False
        self._lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        """Load the model. Call once at startup."""
        if self._loaded:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        """Synchronous model loading (run in thread)."""
        import warnings
        warnings.filterwarnings("ignore")

        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info("Loading SmolVLM-500M-Instruct for local vision...")
        self._processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
        self._model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM-500M-Instruct",
            dtype=torch.float16,
        ).to(self._device)
        self._model.eval()

        vram_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"SmolVLM-500M loaded on {self._device} ({vram_gb:.2f} GB VRAM)")
        self._loaded = True

    def _extract_image_from_messages(self, messages: list[LLMMessage]) -> Optional[Image.Image]:
        """Extract a PIL image from LLM messages (base64 data URL format)."""
        for msg in messages:
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        if url.startswith("data:image/"):
                            # Extract base64 data
                            b64_data = url.split(",", 1)[1] if "," in url else url
                            img_bytes = base64.b64decode(b64_data)
                            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return None

    def _extract_text_from_messages(self, messages: list[LLMMessage]) -> str:
        """Extract the text prompt from messages."""
        for msg in messages:
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
            elif isinstance(msg.content, str) and msg.role == "user":
                return msg.content
        return "Describe this image in one sentence."

    def _infer_sync(self, image: Image.Image, text_prompt: str, max_tokens: int) -> str:
        """Run inference synchronously (called via asyncio.to_thread)."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt},
            ],
        }]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt").to(self._device)

        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=max_tokens)

        result = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        # Extract just the assistant response
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()
        return result

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
    ) -> LLMResponse:
        if not self._loaded:
            await self.load()

        image = self._extract_image_from_messages(messages)
        if image is None:
            return LLMResponse(text="No image provided.", finish_reason="stop")

        text_prompt = self._extract_text_from_messages(messages)

        async with self._lock:  # Prevent concurrent GPU access
            result = await asyncio.to_thread(self._infer_sync, image, text_prompt, max_tokens)

        return LLMResponse(text=result, finish_reason="stop")

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        # Local model doesn't support streaming, just yield the full result
        response = await self.chat(messages, tools, temperature, max_tokens)
        yield response.text
