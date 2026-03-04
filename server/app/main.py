"""Main FastAPI server — WebSocket endpoints for client + UI, REST for auth callbacks."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path for shared imports
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from server.app.config import settings
from server.app.orchestrator import Orchestrator
from server.app.ui.events import ConnectionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
cm = ConnectionManager()
orchestrator = Orchestrator(cm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await orchestrator.initialize()
    logger.info(f"{settings.assistant_name} server started on {settings.server_host}:{settings.server_port}")
    yield
    await orchestrator.shutdown()
    logger.info("Server shut down.")


app = FastAPI(title=f"{settings.assistant_name} Assistant Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the kiosk UI static files
_UI_PATH = _ROOT / "ui" / "kiosk"
if _UI_PATH.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_PATH), html=True), name="kiosk-ui")


# ------------------------------------------------------------------
# WebSocket: Client connection
# ------------------------------------------------------------------

@app.websocket("/ws/client")
async def ws_client(ws: WebSocket):
    await cm.connect_client(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client: {raw[:100]}")
                continue

            event_type = data.get("event", "")
            logger.info(f"Client event: {event_type}")

            handler_map = {
                "wakeword_detected": orchestrator.handle_wakeword,
                "stt_partial": orchestrator.handle_stt_partial,
                "stt_final": orchestrator.handle_stt_final,
                "client_state": orchestrator.handle_client_state,
                "speaker_verified": orchestrator.handle_speaker_verified,
                "user_confirmation": orchestrator.handle_user_confirmation,
            }

            handler = handler_map.get(event_type)
            if handler:
                await handler(data)
            else:
                logger.warning(f"Unknown client event: {event_type}")

    except WebSocketDisconnect:
        cm.disconnect_client(ws)
    except Exception as e:
        logger.error(f"Client WS error: {e}")
        cm.disconnect_client(ws)


# ------------------------------------------------------------------
# WebSocket: UI (Projector) connection
# ------------------------------------------------------------------

@app.websocket("/ws/ui")
async def ws_ui(ws: WebSocket):
    await cm.connect_ui(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event_type = data.get("event", "")

            # UI can send confirmation events and mic control
            if event_type == "user_confirmation":
                await orchestrator.handle_user_confirmation(data)
            elif event_type in ("mic_mute", "mic_unmute", "stop_speaking"):
                logger.info(f"UI → Client: {event_type}")
                await cm.send_to_client(data)
            else:
                logger.info(f"UI event: {event_type}")

    except WebSocketDisconnect:
        cm.disconnect_ui(ws)
    except Exception as e:
        logger.error(f"UI WS error: {e}")
        cm.disconnect_ui(ws)


# ------------------------------------------------------------------
# REST: Health check
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "assistant": settings.assistant_name,
        "clients": cm.client_count,
        "uis": cm.ui_count,
    }


# ------------------------------------------------------------------
# REST: Debug endpoints
# ------------------------------------------------------------------

@app.get("/api/tools")
async def list_tools():
    return [t.model_dump() for t in orchestrator.tools.get_definitions()]


@app.get("/api/memories")
async def list_memories():
    entries = await orchestrator.memory.list_memories(50)
    return [e.model_dump() for e in entries]


@app.get("/api/audit")
async def get_audit_log():
    return await orchestrator.memory.get_audit_log(100)


@app.get("/api/pending")
async def get_pending_actions():
    return [a.model_dump() for a in orchestrator.confirmations.get_all_pending()]


# ------------------------------------------------------------------
# REST: Device management
# ------------------------------------------------------------------

@app.post("/api/devices")
async def register_device(payload: dict):
    from server.app.devices.registry import Device
    device = Device(
        name=payload.get("name", ""),
        aliases=payload.get("aliases", []),
        device_type=payload.get("device_type", "switch"),
        protocol=payload.get("protocol", "http"),
        address=payload.get("address", ""),
        room=payload.get("room", ""),
        config=payload.get("config", {}),
    )
    if not device.name:
        return {"error": "Device name is required"}
    device_id = await orchestrator.device_registry.register(device)
    return {"status": "ok", "device_id": device_id, "name": device.name}


@app.get("/api/devices")
async def list_devices():
    devices = await orchestrator.device_registry.list_all()
    return [d.to_dict() for d in devices]


@app.put("/api/devices/{device_id}")
async def update_device(device_id: int, payload: dict):
    ok = await orchestrator.device_registry.update(device_id, **payload)
    if not ok:
        return {"error": f"Device {device_id} not found or no valid fields to update"}
    return {"status": "ok", "device_id": device_id}


@app.delete("/api/devices/{device_id}")
async def delete_device(device_id: int):
    ok = await orchestrator.device_registry.unregister(device_id)
    if not ok:
        return {"error": f"Device {device_id} not found"}
    return {"status": "ok", "device_id": device_id}


@app.post("/api/devices/discover-ha")
async def discover_ha_devices(payload: dict = {}):
    """Discover controllable entities from Home Assistant and optionally auto-register."""
    domain = payload.get("domain", None)
    auto_register = payload.get("auto_register", False)

    try:
        from server.app.devices.homeassistant_driver import HomeAssistantDriver
        driver = orchestrator.device_controller._get_driver("homeassistant")
        if not isinstance(driver, HomeAssistantDriver):
            return {"error": "Home Assistant driver not available"}
    except Exception as e:
        return {"error": f"Home Assistant not configured: {e}"}

    entities = await driver.discover_entities(domain_filter=domain)
    registered = 0

    if auto_register and entities:
        from server.app.devices.registry import Device
        existing = await orchestrator.device_registry.list_all()
        existing_addresses = {d.address for d in existing}

        for ent in entities:
            eid = ent["entity_id"]
            if eid in existing_addresses:
                continue
            device_type_map = {
                "light": "dimmer", "switch": "switch", "fan": "fan",
                "cover": "motor", "lock": "switch",
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
            await orchestrator.device_registry.register(device)
            registered += 1

    return {"status": "ok", "discovered": len(entities), "registered": registered, "entities": entities[:50]}


# ------------------------------------------------------------------
# REST: Proactive behaviors
# ------------------------------------------------------------------

@app.get("/api/behaviors")
async def list_behaviors():
    if not orchestrator.behavior_manager:
        return {"error": "Behavior system not initialized"}
    return orchestrator.behavior_manager.list_rules()


@app.put("/api/behaviors/{name}")
async def update_behavior(name: str, body: dict):
    if not orchestrator.behavior_manager:
        return {"error": "Behavior system not initialized"}
    rule = orchestrator.behavior_manager.get_rule(name)
    if not rule:
        return {"error": f"Rule '{name}' not found"}
    if "enabled" in body:
        if body["enabled"]:
            orchestrator.behavior_manager.enable_rule(name)
        else:
            orchestrator.behavior_manager.disable_rule(name)
    if "cooldown_seconds" in body:
        rule.cooldown_seconds = float(body["cooldown_seconds"])
    return {"status": "ok", "rule": name, "enabled": rule.enabled}


@app.get("/api/behaviors/log")
async def behavior_log(limit: int = 50):
    if not orchestrator.behavior_manager:
        return {"error": "Behavior system not initialized"}
    return orchestrator.behavior_manager.get_log(limit)


@app.get("/api/events")
async def recent_events(event_type: str = None, limit: int = 50):
    if not orchestrator.event_bus:
        return {"error": "Event system not initialized"}
    events = orchestrator.event_bus.get_recent_events(event_type, limit)
    return [{"type": e.type, "timestamp": e.timestamp, "data": e.data, "source": e.source} for e in events]


# ------------------------------------------------------------------
# REST: Memory browsing
# ------------------------------------------------------------------

@app.get("/api/conversations")
async def list_conversations(limit: int = 20):
    return await orchestrator.memory.list_sessions(limit)


@app.get("/api/conversations/search")
async def search_conversations(q: str = "", limit: int = 20):
    if not q:
        return {"error": "Query parameter 'q' is required"}
    return await orchestrator.memory.search_conversations(q, limit)


@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str):
    turns = await orchestrator.memory.get_session_turns(session_id)
    if not turns:
        return {"error": f"Session '{session_id}' not found"}
    return {"session_id": session_id, "turns": turns}


@app.get("/api/people")
async def list_people():
    people = await orchestrator.memory.list_people()
    return [p.model_dump() for p in people]


@app.get("/api/preferences")
async def list_preferences():
    prefs = await orchestrator.memory.list_all_preferences()
    return [p.model_dump() for p in prefs]


@app.get("/api/facts")
async def list_facts(limit: int = 100):
    facts = await orchestrator.memory.list_all_facts(limit)
    return [f.model_dump() for f in facts]


@app.get("/api/episodes")
async def list_episodes(limit: int = 50):
    episodes = await orchestrator.memory.list_episodes(limit)
    return [e.model_dump() for e in episodes]


# ------------------------------------------------------------------
# REST: Video stream (MJPEG)
# ------------------------------------------------------------------

@app.get("/api/video/stream")
async def video_stream(fps: int = 10):
    """MJPEG live stream from the camera. Use in <img src="...">."""
    camera = orchestrator.camera_manager
    if camera is None:
        return {"error": "Camera not available"}

    fps = max(1, min(fps, 30))

    async def generate():
        try:
            async for jpeg_bytes in camera.stream_frames(fps=fps):
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg_bytes
                    + b"\r\n"
                )
        except Exception as e:
            logger.error(f"Video stream error: {e}")

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/video/context")
async def video_context():
    """Get the current always-on vision context buffer."""
    vc = orchestrator.vision_context
    if vc is None or not vc.is_running:
        return {"status": "disabled", "entries": []}
    context_text = vc.get_recent_context(seconds=60.0, max_entries=20)
    last = vc.get_last_description()
    return {
        "status": "running",
        "last_description": last,
        "context": context_text,
        "buffer_size": len(vc._buffer),
    }


@app.get("/api/video/snapshot")
async def video_snapshot():
    """Single JPEG snapshot from the camera."""
    camera = orchestrator.camera_manager
    if camera is None:
        return {"error": "Camera not available"}
    try:
        jpeg_bytes, cam_id = await camera.capture()
        from fastapi.responses import Response
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Snapshot error: {e}")
        return {"error": str(e)}


# ------------------------------------------------------------------
# OAuth: Google (Gmail + Calendar)
# ------------------------------------------------------------------

@app.get("/auth/google")
async def google_auth_start():
    """Redirect user to Google consent screen."""
    from server.app.auth.google import get_auth_url
    if not settings.google_client_id or not settings.google_client_secret:
        return {"error": "Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env"}
    url = get_auth_url(settings.google_client_id, settings.google_client_secret, settings.google_redirect_uri)
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url)


@app.get("/auth/google/callback")
async def google_auth_callback(code: str = "", error: str = ""):
    """Handle Google OAuth callback — exchange code for tokens."""
    if error:
        return HTMLResponse(f"<h2>Google Auth Error</h2><p>{error}</p><p><a href='/integrations'>Back to setup</a></p>")
    if not code:
        return HTMLResponse("<h2>No auth code received</h2><p><a href='/integrations'>Back to setup</a></p>")

    try:
        from server.app.auth.google import exchange_code
        exchange_code(code, settings.google_client_id, settings.google_client_secret, settings.google_redirect_uri)

        # Reset cached services so they pick up new creds
        from server.app.tools.email_gmail import _reset_service as reset_gmail
        from server.app.tools.calendar_tools import _reset_service as reset_cal
        reset_gmail()
        reset_cal()

        return HTMLResponse(
            "<html><body style='background:#0a0a0f;color:#e0e0e8;font-family:sans-serif;display:flex;"
            "align-items:center;justify-content:center;height:100vh;flex-direction:column'>"
            "<h1 style='color:#22c55e'>Google Connected!</h1>"
            "<p>Gmail and Calendar are now linked to NOVA.</p>"
            "<p><a href='/integrations' style='color:#6366f1'>Back to Integration Setup</a></p>"
            "</body></html>"
        )
    except Exception as e:
        logger.error(f"Google OAuth error: {e}")
        return HTMLResponse(f"<h2>Auth Failed</h2><p>{e}</p><p><a href='/auth/google'>Try again</a></p>")


@app.get("/auth/google/status")
async def google_auth_status():
    """Check if Google OAuth is connected."""
    from server.app.auth.google import is_authenticated
    return {"connected": is_authenticated()}


# ------------------------------------------------------------------
# REST: Integration Setup UI
# ------------------------------------------------------------------

_TEMPLATES_PATH = Path(__file__).parent / "templates"
_ENV_PATH = _ROOT / ".env"


@app.get("/integrations", response_class=HTMLResponse)
async def integrations_page():
    """Serve the Integration Setup UI."""
    html_file = _TEMPLATES_PATH / "integrations.html"
    if not html_file.exists():
        return HTMLResponse("<h1>Integration setup page not found</h1>", status_code=404)
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


@app.get("/api/integrations/env")
async def get_env_keys():
    """Read current .env file and return all non-secret key names + values."""
    if not _ENV_PATH.exists():
        return {"keys": {}}
    keys = {}
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Only return keys that have actual values (not placeholders)
        if value and value != "..." and value != '""' and value != "''":
            keys[key] = value
    return {"keys": keys}


@app.post("/api/integrations/save")
async def save_env_keys(payload: dict):
    """Merge submitted keys into the .env file, preserving comments and structure."""
    new_keys: dict = payload.get("keys", {})
    if not new_keys:
        return {"error": "No keys provided"}

    # Read existing .env
    lines = []
    if _ENV_PATH.exists():
        lines = _ENV_PATH.read_text(encoding="utf-8").splitlines()

    # Build a map of existing key → line index for updating in place
    key_line_map: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            key_line_map[k] = i

    updated = 0
    added = 0
    for key, value in new_keys.items():
        if not key or not value:
            continue
        if key in key_line_map:
            # Update existing line
            lines[key_line_map[key]] = f"{key}={value}"
            updated += 1
        else:
            # Append new key
            lines.append(f"{key}={value}")
            added += 1

    # Write back
    _ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    msg = f"Saved! {updated} updated, {added} added."
    return {"status": "ok", "updated": updated, "added": added, "message": msg}


# ------------------------------------------------------------------
# REST: Manual text input (for testing without mic)
# ------------------------------------------------------------------

@app.post("/api/chat")
async def manual_chat(payload: dict):
    text = payload.get("text", "")
    if not text:
        return {"error": "No text provided"}

    # Capture the assistant's response by temporarily intercepting send_to_client
    captured_responses: list[str] = []
    original_send = cm.send_to_client

    async def intercepting_send(data):
        await original_send(data)
        if isinstance(data, dict) and data.get("event") == "assistant_text":
            captured_responses.append(data.get("text", ""))

    cm.send_to_client = intercepting_send
    try:
        await orchestrator.handle_stt_final({"text": text, "event": "stt_final"})
    finally:
        cm.send_to_client = original_send

    assistant_text = captured_responses[-1] if captured_responses else ""
    return {"status": "ok", "user": text, "assistant": assistant_text}


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server.app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )
