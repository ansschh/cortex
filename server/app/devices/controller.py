"""Device controller — resolves protocol drivers and sends commands to physical devices."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx

from server.app.config import settings

logger = logging.getLogger(__name__)


class BaseProtocolDriver(ABC):
    """Abstract protocol driver for sending commands to devices."""

    @abstractmethod
    async def send_command(
        self, address: str, action: str, params: dict[str, Any], config: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def read_state(self, address: str, config: dict[str, Any]) -> dict[str, Any]:
        ...


class HttpDriver(BaseProtocolDriver):
    """Sends commands to devices over HTTP (e.g. Raspberry Pi endpoint)."""

    def __init__(self, timeout: float = 5.0):
        self._timeout = timeout

    async def send_command(
        self, address: str, action: str, params: dict[str, Any], config: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{address.rstrip('/')}/command"
        payload = {"action": action, "params": params}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def read_state(self, address: str, config: dict[str, Any]) -> dict[str, Any]:
        url = f"{address.rstrip('/')}/state"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()


class GpioDriver(BaseProtocolDriver):
    """Placeholder — direct GPIO control (not yet implemented)."""

    async def send_command(self, address, action, params, config):
        return {"error": "GPIO driver not implemented yet"}

    async def read_state(self, address, config):
        return {"error": "GPIO driver not implemented yet"}


class DeviceController:
    """Resolves protocol → driver, executes commands, returns results."""

    def __init__(self):
        self._http_driver = HttpDriver(timeout=settings.device_controller_timeout)
        self._gpio_driver = GpioDriver()

        # Import real drivers (lazy — only fail when actually used)
        self._mqtt_driver = None
        self._ha_driver = None

        self._drivers: dict[str, BaseProtocolDriver] = {
            "http": self._http_driver,
            "gpio": self._gpio_driver,
        }

    def _get_driver(self, protocol: str) -> BaseProtocolDriver:
        # Lazy-init MQTT driver on first use
        if protocol == "mqtt":
            if self._mqtt_driver is None:
                from server.app.devices.mqtt_driver import MqttDriver
                self._mqtt_driver = MqttDriver()
                self._drivers["mqtt"] = self._mqtt_driver
            return self._mqtt_driver

        # Lazy-init Home Assistant driver on first use
        if protocol == "homeassistant":
            if self._ha_driver is None:
                from server.app.devices.homeassistant_driver import HomeAssistantDriver
                self._ha_driver = HomeAssistantDriver()
                self._drivers["homeassistant"] = self._ha_driver
            return self._ha_driver

        driver = self._drivers.get(protocol)
        if not driver:
            raise ValueError(f"Unknown protocol: {protocol}")
        return driver

    async def send_command(
        self,
        protocol: str,
        address: str,
        action: str,
        params: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        driver = self._get_driver(protocol)
        try:
            return await driver.send_command(address, action, params or {}, config or {})
        except httpx.ConnectError:
            return {"error": f"Cannot connect to device at {address}"}
        except httpx.TimeoutException:
            return {"error": f"Device at {address} timed out"}
        except Exception as e:
            logger.error(f"Device command error ({protocol}://{address}): {e}")
            return {"error": str(e)}

    async def read_state(
        self, protocol: str, address: str, config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        driver = self._get_driver(protocol)
        try:
            return await driver.read_state(address, config or {})
        except httpx.ConnectError:
            return {"error": f"Cannot connect to device at {address}"}
        except httpx.TimeoutException:
            return {"error": f"Device at {address} timed out"}
        except Exception as e:
            logger.error(f"Device state read error ({protocol}://{address}): {e}")
            return {"error": str(e)}

    async def shutdown(self):
        """Clean shutdown of all drivers."""
        if self._mqtt_driver:
            try:
                self._mqtt_driver.disconnect()
            except Exception as e:
                logger.error(f"MQTT shutdown error: {e}")
