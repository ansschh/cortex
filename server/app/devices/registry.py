"""Device registry — SQLite-backed catalog of registered smart home devices."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import aiosqlite


@dataclass
class Device:
    id: int = 0
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    device_type: str = "switch"  # switch | motor | servo | sensor | dimmer | fan
    protocol: str = "http"       # http | gpio | mqtt | serial | homeassistant
    address: str = ""            # e.g. "http://192.168.1.50:5000"
    state: dict[str, Any] = field(default_factory=dict)
    room: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    registered_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "aliases": self.aliases,
            "device_type": self.device_type,
            "protocol": self.protocol,
            "address": self.address,
            "state": self.state,
            "room": self.room,
            "config": self.config,
            "registered_at": self.registered_at,
        }


_DEVICES_SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    aliases TEXT DEFAULT '[]',
    device_type TEXT DEFAULT 'switch',
    protocol TEXT DEFAULT 'http',
    address TEXT DEFAULT '',
    state TEXT DEFAULT '{}',
    room TEXT DEFAULT '',
    config TEXT DEFAULT '{}',
    registered_at TEXT DEFAULT (datetime('now'))
);
"""


class DeviceRegistry:
    """Manages the catalog of registered devices in SQLite."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def initialize(self) -> None:
        """Create the devices table if it doesn't exist."""
        await self._db.executescript(_DEVICES_SCHEMA)
        await self._db.commit()

    async def register(self, device: Device) -> int:
        """Register a new device and return its ID."""
        cur = await self._db.execute(
            """INSERT INTO devices (name, aliases, device_type, protocol, address, state, room, config)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                device.name,
                json.dumps(device.aliases),
                device.device_type,
                device.protocol,
                device.address,
                json.dumps(device.state),
                device.room,
                json.dumps(device.config),
            ),
        )
        await self._db.commit()
        return cur.lastrowid

    async def unregister(self, device_id: int) -> bool:
        """Remove a device by ID. Returns True if it existed."""
        cur = await self._db.execute("DELETE FROM devices WHERE id = ?", (device_id,))
        await self._db.commit()
        return cur.rowcount > 0

    async def get(self, device_id: int) -> Optional[Device]:
        """Get a device by ID."""
        cur = await self._db.execute("SELECT * FROM devices WHERE id = ?", (device_id,))
        row = await cur.fetchone()
        if not row:
            return None
        return self._row_to_device(row)

    async def find_by_name(self, name_or_alias: str) -> Optional[Device]:
        """Fuzzy-search for a device by name or alias (case-insensitive)."""
        query = name_or_alias.lower().strip()

        # First try exact name match
        cur = await self._db.execute(
            "SELECT * FROM devices WHERE LOWER(name) = ?", (query,)
        )
        row = await cur.fetchone()
        if row:
            return self._row_to_device(row)

        # Then try LIKE on name
        cur = await self._db.execute(
            "SELECT * FROM devices WHERE LOWER(name) LIKE ?", (f"%{query}%",)
        )
        row = await cur.fetchone()
        if row:
            return self._row_to_device(row)

        # Scan aliases JSON array
        cur = await self._db.execute("SELECT * FROM devices")
        rows = await cur.fetchall()
        for r in rows:
            aliases = json.loads(r["aliases"] or "[]")
            for alias in aliases:
                if query in alias.lower():
                    return self._row_to_device(r)

        return None

    async def list_all(self, room: Optional[str] = None) -> list[Device]:
        """List all devices, optionally filtered by room."""
        if room:
            cur = await self._db.execute(
                "SELECT * FROM devices WHERE LOWER(room) = ? ORDER BY name",
                (room.lower(),),
            )
        else:
            cur = await self._db.execute("SELECT * FROM devices ORDER BY name")
        rows = await cur.fetchall()
        return [self._row_to_device(r) for r in rows]

    async def update_state(self, device_id: int, state: dict) -> None:
        """Update the cached state of a device."""
        await self._db.execute(
            "UPDATE devices SET state = ? WHERE id = ?",
            (json.dumps(state), device_id),
        )
        await self._db.commit()

    async def update(self, device_id: int, **fields) -> bool:
        """Update arbitrary fields on a device."""
        allowed = {"name", "aliases", "device_type", "protocol", "address", "room", "config"}
        sets = []
        values = []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k in ("aliases", "config"):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            values.append(v)
        if not sets:
            return False
        values.append(device_id)
        cur = await self._db.execute(
            f"UPDATE devices SET {', '.join(sets)} WHERE id = ?", values
        )
        await self._db.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_device(row) -> Device:
        return Device(
            id=row["id"],
            name=row["name"],
            aliases=json.loads(row["aliases"] or "[]"),
            device_type=row["device_type"],
            protocol=row["protocol"],
            address=row["address"],
            state=json.loads(row["state"] or "{}"),
            room=row["room"],
            config=json.loads(row["config"] or "{}"),
            registered_at=row["registered_at"],
        )
