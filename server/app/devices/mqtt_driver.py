"""MQTT device driver — paho-mqtt v2, background loop, Tasmota support."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

from server.app.config import settings
from server.app.devices.controller import BaseProtocolDriver

logger = logging.getLogger(__name__)


class MqttDriver(BaseProtocolDriver):
    """Real MQTT driver for controlling devices over MQTT brokers.

    Features:
    - Lazy connection on first use
    - Background loop thread (paho-mqtt v2 style)
    - Topic patterns: {prefix}/{device_name}/command (publish), {prefix}/{device_name}/state (subscribe)
    - Tasmota compatibility mode via device config: {"device_style": "tasmota"}
    - State cache from subscribed messages
    """

    def __init__(
        self,
        broker_host: str = "",
        broker_port: int = 1883,
        username: str = "",
        password: str = "",
        topic_prefix: str = "nova",
    ):
        self._broker_host = broker_host or getattr(settings, "mqtt_broker_host", "localhost")
        self._broker_port = broker_port or getattr(settings, "mqtt_broker_port", 1883)
        self._username = username or getattr(settings, "mqtt_username", "")
        self._password = password or getattr(settings, "mqtt_password", "")
        self._topic_prefix = topic_prefix or getattr(settings, "mqtt_topic_prefix", "nova")

        self._client = None
        self._connected = False
        self._lock = threading.Lock()

        # State cache: topic → last payload
        self._state_cache: dict[str, dict[str, Any]] = {}

    def _ensure_connected(self):
        """Lazy connect to the MQTT broker on first use."""
        if self._connected and self._client:
            return

        with self._lock:
            if self._connected and self._client:
                return

            try:
                import paho.mqtt.client as mqtt
            except ImportError:
                raise RuntimeError(
                    "paho-mqtt not installed. Run: pip install paho-mqtt>=2.0.0"
                )

            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"nova-{int(time.time())}",
            )

            if self._username:
                client.username_pw_set(self._username, self._password)

            client.on_connect = self._on_connect
            client.on_message = self._on_message
            client.on_disconnect = self._on_disconnect

            try:
                client.connect(self._broker_host, self._broker_port, keepalive=60)
                client.loop_start()  # Background thread
                self._client = client
                self._connected = True
                logger.info(f"MQTT connected to {self._broker_host}:{self._broker_port}")
            except Exception as e:
                logger.error(f"MQTT connection failed: {e}")
                raise

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        """Subscribe to state topics on connect."""
        logger.info(f"MQTT connected (rc={reason_code})")
        # Subscribe to all state topics under our prefix
        topic = f"{self._topic_prefix}/+/state"
        client.subscribe(topic)
        # Also subscribe to Tasmota-style stat topics
        client.subscribe("stat/+/RESULT")
        client.subscribe("stat/+/STATE")
        client.subscribe("tele/+/STATE")

    def _on_message(self, client, userdata, msg):
        """Cache incoming state messages."""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"raw": msg.payload.decode("utf-8", errors="replace")}

        self._state_cache[msg.topic] = payload
        logger.debug(f"MQTT state update: {msg.topic} → {str(payload)[:200]}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        logger.warning(f"MQTT disconnected (rc={reason_code})")
        self._connected = False

    def _build_topic(self, address: str, suffix: str, config: dict) -> str:
        """Build MQTT topic from address and device style.

        For standard devices: {prefix}/{address}/{suffix}
        For Tasmota: cmnd/{address}/{command}
        """
        device_style = config.get("device_style", "standard")
        if device_style == "tasmota":
            return f"cmnd/{address}/{suffix}"
        return f"{self._topic_prefix}/{address}/{suffix}"

    def _build_state_topic(self, address: str, config: dict) -> str:
        """Build the state topic to check cache."""
        device_style = config.get("device_style", "standard")
        if device_style == "tasmota":
            return f"stat/{address}/RESULT"
        return f"{self._topic_prefix}/{address}/state"

    async def send_command(
        self, address: str, action: str, params: dict[str, Any], config: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish a command to the device's MQTT topic."""
        try:
            self._ensure_connected()
        except Exception as e:
            return {"error": f"MQTT connection failed: {e}"}

        device_style = config.get("device_style", "standard")

        if device_style == "tasmota":
            return self._send_tasmota(address, action, params)
        else:
            return self._send_standard(address, action, params, config)

    def _send_standard(self, address: str, action: str, params: dict, config: dict) -> dict:
        """Standard MQTT: publish JSON payload to {prefix}/{address}/command."""
        topic = self._build_topic(address, "command", config)
        payload = json.dumps({"action": action, "params": params})

        result = self._client.publish(topic, payload, qos=1)
        if result.rc == 0:
            return {"status": "published", "topic": topic, "action": action}
        return {"error": f"MQTT publish failed (rc={result.rc})"}

    def _send_tasmota(self, address: str, action: str, params: dict) -> dict:
        """Tasmota-style: publish to cmnd/{address}/Power etc."""
        # Map common actions to Tasmota commands
        tasmota_map = {
            "on": ("Power", "ON"),
            "off": ("Power", "OFF"),
            "toggle": ("Power", "TOGGLE"),
        }

        if action in tasmota_map:
            cmd, value = tasmota_map[action]
        elif action == "set":
            # Handle dimming, color, etc.
            if "brightness" in params:
                cmd, value = "Dimmer", str(params["brightness"])
            elif "color" in params:
                cmd, value = "Color", str(params["color"])
            elif "speed" in params:
                cmd, value = "Speed", str(params["speed"])
            else:
                cmd, value = "Power", "ON"
        else:
            cmd, value = action, json.dumps(params) if params else ""

        topic = f"cmnd/{address}/{cmd}"
        result = self._client.publish(topic, value, qos=1)
        if result.rc == 0:
            return {"status": "published", "topic": topic, "command": cmd, "value": value}
        return {"error": f"MQTT publish failed (rc={result.rc})"}

    async def read_state(self, address: str, config: dict[str, Any]) -> dict[str, Any]:
        """Read the last cached state for a device, or request fresh state."""
        try:
            self._ensure_connected()
        except Exception as e:
            return {"error": f"MQTT connection failed: {e}"}

        state_topic = self._build_state_topic(address, config)

        # Check cache first
        if state_topic in self._state_cache:
            return self._state_cache[state_topic]

        # Request fresh state (Tasmota: send empty to cmnd/device/State)
        device_style = config.get("device_style", "standard")
        if device_style == "tasmota":
            self._client.publish(f"cmnd/{address}/State", "", qos=1)
        else:
            self._client.publish(
                f"{self._topic_prefix}/{address}/state/get", "", qos=1,
            )

        # Wait briefly for response
        import asyncio
        await asyncio.sleep(0.5)

        if state_topic in self._state_cache:
            return self._state_cache[state_topic]

        return {"status": "unknown", "note": "No state response received"}

    def disconnect(self):
        """Clean disconnect from MQTT broker."""
        if self._client and self._connected:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception as e:
                logger.error(f"MQTT disconnect error: {e}")
            self._connected = False
            logger.info("MQTT disconnected")
