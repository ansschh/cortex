"""WebSocket event manager — broadcasts events to connected UI and client sockets."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for both client and UI."""

    def __init__(self):
        self._client_connections: list[WebSocket] = []
        self._ui_connections: list[WebSocket] = []

    async def connect_client(self, ws: WebSocket) -> None:
        await ws.accept()
        self._client_connections.append(ws)
        logger.info(f"Client connected. Total clients: {len(self._client_connections)}")

    async def connect_ui(self, ws: WebSocket) -> None:
        await ws.accept()
        self._ui_connections.append(ws)
        logger.info(f"UI connected. Total UIs: {len(self._ui_connections)}")

    def disconnect_client(self, ws: WebSocket) -> None:
        if ws in self._client_connections:
            self._client_connections.remove(ws)
        logger.info(f"Client disconnected. Total clients: {len(self._client_connections)}")

    def disconnect_ui(self, ws: WebSocket) -> None:
        if ws in self._ui_connections:
            self._ui_connections.remove(ws)
        logger.info(f"UI disconnected. Total UIs: {len(self._ui_connections)}")

    async def send_to_client(self, data: dict[str, Any]) -> None:
        dead = []
        for ws in self._client_connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect_client(ws)

    async def send_to_ui(self, data: dict[str, Any]) -> None:
        dead = []
        for ws in self._ui_connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect_ui(ws)

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send to both client and UI connections."""
        await asyncio.gather(
            self.send_to_client(data),
            self.send_to_ui(data),
        )

    @property
    def client_count(self) -> int:
        return len(self._client_connections)

    @property
    def ui_count(self) -> int:
        return len(self._ui_connections)
