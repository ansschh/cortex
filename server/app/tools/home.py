"""Real home control tools — device registry + HTTP-based device controller."""

from __future__ import annotations

import json
import logging
from typing import Any

from server.app.devices.controller import DeviceController
from server.app.devices.registry import Device, DeviceRegistry
from server.app.tools.base import BaseTool
from shared.schemas.events import SensitivityLevel
from shared.schemas.tool_calls import ToolResult

logger = logging.getLogger(__name__)


class HomeCommandTool(BaseTool):
    name = "home.command"
    description = (
        "Send a command to a home device (lights, fan, etc.). "
        "Accepts a natural-language device name (e.g. 'sink light') and fuzzy-matches it against registered devices."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "device": {
                "type": "string",
                "description": "Natural language device name (e.g. 'sink light', 'wardrobe light', 'desk lamp')",
            },
            "action": {
                "type": "string",
                "enum": ["on", "off", "toggle", "set"],
                "description": "Action to perform",
            },
            "params": {
                "type": "object",
                "description": "Additional parameters (e.g. brightness, speed, color)",
                "default": {},
            },
        },
        "required": ["device", "action"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, registry: DeviceRegistry, controller: DeviceController):
        self._registry = registry
        self._controller = controller

    async def execute(self, **kwargs: Any) -> ToolResult:
        device_name = kwargs.get("device", "")
        action = kwargs.get("action", "")
        params = kwargs.get("params", {})

        # Fuzzy-find the device
        device = await self._registry.find_by_name(device_name)
        if not device:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"No device found matching '{device_name}'. Use home.list_devices to see available devices.",
            )

        # Send command via the appropriate protocol driver
        result = await self._controller.send_command(
            device.protocol, device.address, action, params, device.config,
        )

        if "error" in result:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"Device '{device.name}' command failed: {result['error']}",
            )

        # Update cached state
        new_state = device.state.copy()
        new_state["last_action"] = action
        new_state.update(result.get("state", {}))
        await self._registry.update_state(device.id, new_state)

        return ToolResult(
            tool_name=self.name, success=True,
            result={"device": device.name, "action": action, "response": result},
            display_card={
                "card_type": "DeviceStatusCard",
                "title": f"Device: {device.name}",
                "body": {"action": action, "status": "ok", "response": result},
            },
        )


class HomeListDevicesTool(BaseTool):
    name = "home.list_devices"
    description = "List all registered home devices and their current state."
    parameters_schema = {
        "type": "object",
        "properties": {
            "room": {
                "type": "string",
                "description": "Optional room filter",
            },
        },
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, registry: DeviceRegistry):
        self._registry = registry

    async def execute(self, **kwargs: Any) -> ToolResult:
        room = kwargs.get("room")
        devices = await self._registry.list_all(room=room)
        device_list = [d.to_dict() for d in devices]
        return ToolResult(
            tool_name=self.name, success=True, result=device_list,
            display_card={
                "card_type": "DeviceStatusCard",
                "title": "Home Devices",
                "body": device_list,
            },
        )


class HomeRegisterDeviceTool(BaseTool):
    name = "home.register_device"
    description = "Register a new smart home device (name, aliases, type, protocol, address, room)."
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Device name (e.g. 'Sink Light')"},
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Alternative names for the device",
                "default": [],
            },
            "device_type": {
                "type": "string",
                "enum": ["switch", "motor", "servo", "sensor", "dimmer", "fan"],
                "description": "Type of device",
                "default": "switch",
            },
            "protocol": {
                "type": "string",
                "enum": ["http", "gpio", "mqtt", "serial", "homeassistant"],
                "description": "Communication protocol",
                "default": "http",
            },
            "address": {"type": "string", "description": "Device address (e.g. 'http://192.168.1.50:5000')"},
            "room": {"type": "string", "description": "Room the device is in", "default": ""},
        },
        "required": ["name", "address"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, registry: DeviceRegistry):
        self._registry = registry

    async def execute(self, **kwargs: Any) -> ToolResult:
        device = Device(
            name=kwargs.get("name", ""),
            aliases=kwargs.get("aliases", []),
            device_type=kwargs.get("device_type", "switch"),
            protocol=kwargs.get("protocol", "http"),
            address=kwargs.get("address", ""),
            room=kwargs.get("room", ""),
        )

        device_id = await self._registry.register(device)
        return ToolResult(
            tool_name=self.name, success=True,
            result={"device_id": device_id, "name": device.name, "address": device.address},
            display_card={
                "card_type": "DeviceStatusCard",
                "title": f"Registered: {device.name}",
                "body": {"id": device_id, "name": device.name, "type": device.device_type,
                         "protocol": device.protocol, "address": device.address},
            },
        )


class HomeDiscoverHATool(BaseTool):
    """Discover controllable entities from a connected Home Assistant instance."""
    name = "home.discover_ha"
    description = (
        "Discover all controllable devices from Home Assistant and optionally "
        "auto-register them. Requires HA_URL and HA_TOKEN to be configured."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "Filter by HA domain (light, switch, fan, cover, lock, climate, media_player). Leave empty for all.",
                "default": "",
            },
            "auto_register": {
                "type": "boolean",
                "description": "If true, automatically register discovered entities as NOVA devices.",
                "default": False,
            },
        },
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, registry: DeviceRegistry, controller: DeviceController):
        self._registry = registry
        self._controller = controller

    async def execute(self, **kwargs: Any) -> ToolResult:
        domain = kwargs.get("domain", "") or None
        auto_register = kwargs.get("auto_register", False)

        # Get the HA driver
        try:
            from server.app.devices.homeassistant_driver import HomeAssistantDriver
            driver = self._controller._get_driver("homeassistant")
            if not isinstance(driver, HomeAssistantDriver):
                return ToolResult(tool_name=self.name, success=False, error="Home Assistant driver not available")
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"Home Assistant not configured: {e}")

        entities = await driver.discover_entities(domain_filter=domain)
        if not entities:
            return ToolResult(
                tool_name=self.name, success=True,
                result={"discovered": 0, "message": "No controllable entities found. Check HA_URL and HA_TOKEN."},
            )

        registered = 0
        if auto_register:
            existing = await self._registry.list_all()
            existing_addresses = {d.address for d in existing}

            for ent in entities:
                eid = ent["entity_id"]
                if eid in existing_addresses:
                    continue  # Already registered

                device_type_map = {
                    "light": "dimmer", "switch": "switch", "fan": "fan",
                    "cover": "motor", "lock": "switch", "climate": "sensor",
                    "media_player": "switch",
                }
                device = Device(
                    name=ent.get("friendly_name", eid),
                    aliases=[eid],
                    device_type=device_type_map.get(ent["domain"], "switch"),
                    protocol="homeassistant",
                    address=eid,
                    room="",
                    config={"ha_domain": ent["domain"]},
                )
                await self._registry.register(device)
                registered += 1

        return ToolResult(
            tool_name=self.name, success=True,
            result={
                "discovered": len(entities),
                "registered": registered,
                "entities": entities[:20],  # Cap to avoid huge responses
            },
            display_card={
                "card_type": "DeviceStatusCard",
                "title": "HA Discovery",
                "body": {"found": len(entities), "registered": registered},
            },
        )


class HomeGetDeviceStateTool(BaseTool):
    name = "home.get_device_state"
    description = "Query the current state of a device directly from the device via its protocol driver."
    parameters_schema = {
        "type": "object",
        "properties": {
            "device": {
                "type": "string",
                "description": "Device name or alias to query",
            },
        },
        "required": ["device"],
    }
    requires_confirmation = False
    sensitivity = SensitivityLevel.LOW

    def __init__(self, registry: DeviceRegistry, controller: DeviceController):
        self._registry = registry
        self._controller = controller

    async def execute(self, **kwargs: Any) -> ToolResult:
        device_name = kwargs.get("device", "")
        device = await self._registry.find_by_name(device_name)
        if not device:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"No device found matching '{device_name}'.",
            )

        state = await self._controller.read_state(
            device.protocol, device.address, device.config,
        )

        if "error" in state:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"Failed to read state of '{device.name}': {state['error']}",
            )

        # Cache the state
        await self._registry.update_state(device.id, state)

        return ToolResult(
            tool_name=self.name, success=True,
            result={"device": device.name, "state": state},
            display_card={
                "card_type": "DeviceStatusCard",
                "title": f"State: {device.name}",
                "body": state,
            },
        )
