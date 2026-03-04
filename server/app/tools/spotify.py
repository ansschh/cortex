"""Spotify tools — play music, search, control playback.

Requires Spotify Premium + a registered app at developer.spotify.com.
Uses spotipy library if available, falls back to direct HTTP API calls.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from server.app.tools.base import BaseTool
from shared.schemas.events import SensitivityLevel
from shared.schemas.tool_calls import ToolResult

logger = logging.getLogger(__name__)

_SPOTIFY_API = "https://api.spotify.com/v1"


class _SpotifyBase(BaseTool):
    """Shared Spotify API client logic."""

    def __init__(self, access_token: str = ""):
        self._token = access_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    async def _api_get(self, path: str, params: dict = None) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_SPOTIFY_API}{path}", headers=self._headers(), params=params)
            resp.raise_for_status()
            return resp.json()

    async def _api_put(self, path: str, json: dict = None, params: dict = None) -> int:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.put(f"{_SPOTIFY_API}{path}", headers=self._headers(), json=json, params=params)
            return resp.status_code

    async def _api_post(self, path: str, json: dict = None) -> int:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{_SPOTIFY_API}{path}", headers=self._headers(), json=json)
            return resp.status_code

    def _check_token(self) -> Optional[ToolResult]:
        if not self._token:
            return ToolResult(
                tool_name=self.name, success=False,
                result={"error": "Spotify not connected. Set up OAuth at /auth/spotify"},
            )
        return None


class SpotifyPlayTool(_SpotifyBase):
    name = "spotify.play"
    description = "Play a song, artist, album, or playlist on Spotify. Searches if no URI provided."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to play (song name, artist, playlist, etc.)"},
            "type": {"type": "string", "enum": ["track", "artist", "album", "playlist"], "description": "Type to search for (default: track)"},
        },
        "required": ["query"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.MEDIUM

    async def execute(self, **kwargs: Any) -> ToolResult:
        err = self._check_token()
        if err:
            return err

        query = kwargs["query"]
        search_type = kwargs.get("type", "track")

        try:
            # Search for the item
            data = await self._api_get("/search", params={"q": query, "type": search_type, "limit": 1})
            items_key = f"{search_type}s"
            items = data.get(items_key, {}).get("items", [])

            if not items:
                return ToolResult(tool_name=self.name, success=False, result={"error": f"No {search_type} found for '{query}'"})

            item = items[0]
            uri = item["uri"]
            name = item.get("name", query)

            # Start playback
            if search_type == "track":
                status = await self._api_put("/me/player/play", json={"uris": [uri]})
            else:
                status = await self._api_put("/me/player/play", json={"context_uri": uri})

            if status in (204, 200):
                artist = item.get("artists", [{}])[0].get("name", "") if search_type == "track" else ""
                return ToolResult(
                    tool_name=self.name, success=True,
                    result={"playing": name, "artist": artist, "type": search_type, "uri": uri},
                )
            else:
                return ToolResult(tool_name=self.name, success=False, result={"error": f"Playback failed (status {status}). Is Spotify open on a device?"})

        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, result={"error": str(e)})


class SpotifyPauseTool(_SpotifyBase):
    name = "spotify.pause"
    description = "Pause Spotify playback."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        err = self._check_token()
        if err:
            return err
        try:
            status = await self._api_put("/me/player/pause")
            return ToolResult(tool_name=self.name, success=status in (204, 200), result={"action": "paused"})
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, result={"error": str(e)})


class SpotifySkipTool(_SpotifyBase):
    name = "spotify.skip"
    description = "Skip to the next track."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        err = self._check_token()
        if err:
            return err
        try:
            status = await self._api_post("/me/player/next")
            return ToolResult(tool_name=self.name, success=status in (204, 200), result={"action": "skipped"})
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, result={"error": str(e)})


class SpotifyQueueTool(_SpotifyBase):
    name = "spotify.queue"
    description = "Add a song to the playback queue."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Song to add to queue"},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        err = self._check_token()
        if err:
            return err

        query = kwargs["query"]
        try:
            data = await self._api_get("/search", params={"q": query, "type": "track", "limit": 1})
            tracks = data.get("tracks", {}).get("items", [])
            if not tracks:
                return ToolResult(tool_name=self.name, success=False, result={"error": f"No track found for '{query}'"})

            track = tracks[0]
            status = await self._api_post(f"/me/player/queue?uri={track['uri']}")
            return ToolResult(
                tool_name=self.name, success=status in (204, 200),
                result={"queued": track["name"], "artist": track["artists"][0]["name"]},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, result={"error": str(e)})


class SpotifyNowPlayingTool(_SpotifyBase):
    name = "spotify.now_playing"
    description = "Show the currently playing track."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        err = self._check_token()
        if err:
            return err

        try:
            data = await self._api_get("/me/player/currently-playing")
            if not data or not data.get("item"):
                return ToolResult(tool_name=self.name, success=True, result={"status": "Nothing playing"})

            item = data["item"]
            return ToolResult(
                tool_name=self.name, success=True,
                result={
                    "track": item["name"],
                    "artist": item["artists"][0]["name"],
                    "album": item["album"]["name"],
                    "progress_ms": data.get("progress_ms", 0),
                    "duration_ms": item.get("duration_ms", 0),
                    "is_playing": data.get("is_playing", False),
                },
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, result={"error": str(e)})


class SpotifySearchTool(_SpotifyBase):
    name = "spotify.search"
    description = "Search Spotify for tracks, artists, albums, or playlists."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "type": {"type": "string", "enum": ["track", "artist", "album", "playlist"], "description": "Search type (default: track)"},
            "limit": {"type": "integer", "description": "Number of results (default: 5)"},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        err = self._check_token()
        if err:
            return err

        query = kwargs["query"]
        search_type = kwargs.get("type", "track")
        limit = min(int(kwargs.get("limit", 5)), 10)

        try:
            data = await self._api_get("/search", params={"q": query, "type": search_type, "limit": limit})
            items = data.get(f"{search_type}s", {}).get("items", [])

            results = []
            for item in items:
                entry = {"name": item["name"], "uri": item["uri"]}
                if search_type == "track":
                    entry["artist"] = item["artists"][0]["name"]
                    entry["album"] = item["album"]["name"]
                elif search_type == "artist":
                    entry["genres"] = item.get("genres", [])[:3]
                results.append(entry)

            return ToolResult(tool_name=self.name, success=True, result={"results": results, "count": len(results)})
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, result={"error": str(e)})


class SpotifyVolumeTool(_SpotifyBase):
    name = "spotify.volume"
    description = "Set Spotify playback volume (0-100)."
    parameters_schema = {
        "type": "object",
        "properties": {
            "volume": {"type": "integer", "description": "Volume level 0-100"},
        },
        "required": ["volume"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        err = self._check_token()
        if err:
            return err

        volume = max(0, min(100, int(kwargs["volume"])))
        try:
            status = await self._api_put(f"/me/player/volume?volume_percent={volume}")
            return ToolResult(tool_name=self.name, success=status in (204, 200), result={"volume": volume})
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, result={"error": str(e)})


class SpotifyPlaylistTool(_SpotifyBase):
    name = "spotify.playlist"
    description = "List your Spotify playlists."
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max playlists to return (default: 10)"},
        },
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        err = self._check_token()
        if err:
            return err

        limit = min(int(kwargs.get("limit", 10)), 20)
        try:
            data = await self._api_get("/me/playlists", params={"limit": limit})
            playlists = [
                {"name": p["name"], "tracks": p["tracks"]["total"], "uri": p["uri"]}
                for p in data.get("items", [])
            ]
            return ToolResult(tool_name=self.name, success=True, result={"playlists": playlists, "count": len(playlists)})
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, result={"error": str(e)})
