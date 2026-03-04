"""Home Assistant device driver — REST API integration."""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from server.app.config import settings
from server.app.devices.controller import BaseProtocolDriver

logger = logging.getLogger(__name__)


class HomeAssistantDriver(BaseProtocolDriver):
    """Controls devices through Home Assistant's REST API.

    The device `address` field stores the HA entity_id (e.g., "light.bedroom").
    Auth is via long-lived access token (HA_URL, HA_TOKEN in settings).
    """

    # Map NOVA actions to HA service calls
    _ACTION_MAP = {
        "on": "turn_on",
        "off": "turn_off",
        "toggle": "toggle",
    }

    def __init__(
        self,
        ha_url: str = "",
        ha_token: str = "",
        timeout: float = 10.0,
    ):
        self._ha_url = (ha_url or getattr(settings, "ha_url", "")).rstrip("/")
        self._ha_token = ha_token or getattr(settings, "ha_token", "")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._ha_token}",
            "Content-Type": "application/json",
        }

    def _get_domain(self, entity_id: str) -> str:
        """Extract the domain from an entity_id (e.g., 'light' from 'light.bedroom')."""
        return entity_id.split(".")[0] if "." in entity_id else "switch"

    async def send_command(
        self, address: str, action: str, params: dict[str, Any], config: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a command to a Home Assistant entity.

        Args:
            address: The HA entity_id (e.g., "light.bedroom")
            action: One of "on", "off", "toggle", "set"
            params: Additional params (brightness, color_temp, etc.)
            config: Device config dict
        """
        if not self._ha_url or not self._ha_token:
            return {"error": "Home Assistant not configured. Set HA_URL and HA_TOKEN in .env"}

        entity_id = address
        domain = self._get_domain(entity_id)

        # Map action to HA service
        if action == "set":
            service = "turn_on"  # HA uses turn_on with extra params for set operations
        elif action in self._ACTION_MAP:
            service = self._ACTION_MAP[action]
        else:
            service = action  # Pass through custom service names

        url = f"{self._ha_url}/api/services/{domain}/{service}"
        payload: dict[str, Any] = {"entity_id": entity_id}

        # Map NOVA params to HA service data
        if "brightness" in params:
            # NOVA uses 0-100, HA uses 0-255
            payload["brightness"] = int(params["brightness"] * 255 / 100)
        if "color_temp" in params:
            payload["color_temp"] = params["color_temp"]
        if "color" in params:
            # Accept hex color or RGB list
            color = params["color"]
            if isinstance(color, str) and color.startswith("#"):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                payload["rgb_color"] = [r, g, b]
            elif isinstance(color, list):
                payload["rgb_color"] = color
        if "temperature" in params:
            payload["temperature"] = params["temperature"]
        if "speed" in params:
            payload["percentage"] = params["speed"]  # Fan speed as percentage

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                return {
                    "status": "ok",
                    "entity_id": entity_id,
                    "service": f"{domain}.{service}",
                    "params": payload,
                }
        except httpx.ConnectError:
            return {"error": f"Cannot connect to Home Assistant at {self._ha_url}"}
        except httpx.TimeoutException:
            return {"error": "Home Assistant request timed out"}
        except httpx.HTTPStatusError as e:
            return {"error": f"Home Assistant error: {e.response.status_code} {e.response.text[:200]}"}
        except Exception as e:
            logger.error(f"Home Assistant command error: {e}")
            return {"error": str(e)}

    async def read_state(self, address: str, config: dict[str, Any]) -> dict[str, Any]:
        """Read the current state of a Home Assistant entity."""
        if not self._ha_url or not self._ha_token:
            return {"error": "Home Assistant not configured. Set HA_URL and HA_TOKEN in .env"}

        entity_id = address
        url = f"{self._ha_url}/api/states/{entity_id}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                return {
                    "entity_id": data.get("entity_id", entity_id),
                    "state": data.get("state", "unknown"),
                    "attributes": data.get("attributes", {}),
                    "last_changed": data.get("last_changed", ""),
                }
        except httpx.ConnectError:
            return {"error": f"Cannot connect to Home Assistant at {self._ha_url}"}
        except httpx.TimeoutException:
            return {"error": "Home Assistant request timed out"}
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"error": f"Entity '{entity_id}' not found in Home Assistant"}
            return {"error": f"Home Assistant error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Home Assistant state read error: {e}")
            return {"error": str(e)}

    async def discover_entities(self, domain_filter: str | None = None) -> list[dict[str, Any]]:
        """Pull all controllable entities from Home Assistant.

        Args:
            domain_filter: Optional filter like "light", "switch", "fan", etc.

        Returns:
            List of entity dicts with entity_id, friendly_name, domain, state.
        """
        if not self._ha_url or not self._ha_token:
            return []

        url = f"{self._ha_url}/api/states"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                states = resp.json()
        except Exception as e:
            logger.error(f"Home Assistant discovery error: {e}")
            return []

        # Controllable domains
        controllable = {"light", "switch", "fan", "cover", "lock", "climate", "media_player", "scene", "script"}
        entities = []

        for state in states:
            eid = state.get("entity_id", "")
            domain = eid.split(".")[0] if "." in eid else ""

            if domain not in controllable:
                continue
            if domain_filter and domain != domain_filter:
                continue

            attrs = state.get("attributes", {})
            entities.append({
                "entity_id": eid,
                "friendly_name": attrs.get("friendly_name", eid),
                "domain": domain,
                "state": state.get("state", "unknown"),
                "device_class": attrs.get("device_class", ""),
            })

        logger.info(f"HA discovery: found {len(entities)} controllable entities")
        return entities
