# NOVA — Dorm Room AI Assistant

A local-first, voice-controlled AI assistant with real-time STT/TTS, memory, email/Slack/calendar integration, smart home control (MQTT + Home Assistant), and a projector kiosk UI.

## Architecture

```
Client (Laptop/Desktop)           Server (Orchestrator)              UI (Projector/Browser)
┌────────────────────┐            ┌─────────────────────────┐        ┌───────────────┐
│ Wake Word (OWW)    │            │ Intent Router (LLM)     │        │ Kiosk Cards   │
│ STT (Whisper GPU)  │─── WS ────│ FastPath (JointBERT NLU)│── WS ──│ Status Bar    │
│ TTS (ElevenLabs)   │            │ LLM (Groq/OpenRouter)   │        │ MJPEG Camera  │
│ Speaker Verify     │            │ Memory (SQLite + FAISS)  │        └───────────────┘
│ VAD + Turn Detect  │            │ 60+ Tools               │
│ Local TTS (Piper)  │            │ Policy Gates             │
└────────────────────┘            │ IoT (HTTP/MQTT/HA)       │
                                  │ Vision (SmolVLM)         │
                                  │ Proactive Behaviors      │
                                  └─────────────────────────┘
```

## Quick Start

### Option A: Automated Setup

```bash
# Linux/macOS
chmod +x setup.sh && ./setup.sh

# Windows
setup.bat
```

### Option B: Manual Setup

```bash
# 1. Create venv (Python 3.11 or 3.12 — NOT 3.13)
python3.11 -m venv .venv3
source .venv3/bin/activate  # Linux/macOS
# .venv3\Scripts\activate   # Windows

# 2. Install PyTorch (GPU)
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Or CPU-only: pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install dependencies
pip install -r server/requirements.txt
pip install -r client/requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — add at minimum: GROQ_API_KEY, ELEVEN_API_KEY, ELEVEN_VOICE_ID

# 5. Download models
python scripts/download_models.py

# 6. Run
python run_server.py    # Terminal 1
python run_client.py    # Terminal 2
# Open http://localhost:8000/ui/ for the kiosk dashboard
```

### Option C: Docker (Server Only)

```bash
cp .env.example .env
# Edit .env with your API keys

docker compose up server
# Server available at http://localhost:8000
# Test: curl http://localhost:8000/health
```

## API Keys

| Service | Required | Free Tier | Get Key |
|---------|----------|-----------|---------|
| Groq | Yes | 100k tokens/day | [console.groq.com](https://console.groq.com) |
| ElevenLabs | Yes (voice) | 10k chars/month | [elevenlabs.io](https://elevenlabs.io) |
| OpenWeatherMap | Optional | 1000 calls/day | [openweathermap.org](https://openweathermap.org/api) |
| Google (Gmail/Calendar) | Optional | Free | [console.cloud.google.com](https://console.cloud.google.com) |
| Spotify | Optional | Free | [developer.spotify.com](https://developer.spotify.com) |
| Slack | Optional | Free | [api.slack.com](https://api.slack.com) |

## Integration Setup

### Gmail & Calendar

```bash
# 1. Create OAuth credentials at Google Cloud Console
# 2. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to .env
# 3. Visit http://localhost:8000/auth/google to authorize
```

### Home Assistant

```bash
# 1. Generate a long-lived access token in HA (Profile → Security → Long-Lived Access Tokens)
# 2. Add to .env:
#    HA_URL=http://homeassistant.local:8123
#    HA_TOKEN=your_token_here
# 3. Discover devices: "Hey Nova, discover my Home Assistant devices"
#    Or: POST http://localhost:8000/api/devices/discover-ha {"auto_register": true}
```

### MQTT Devices

```bash
# 1. Add to .env:
#    MQTT_BROKER_HOST=your_broker_ip
#    MQTT_BROKER_PORT=1883
#    MQTT_USERNAME=your_username  (optional)
#    MQTT_PASSWORD=your_password  (optional)
# 2. Register devices via voice or API:
#    "Hey Nova, register a device called desk lamp, protocol mqtt, address desk_lamp"
#    Or: POST http://localhost:8000/api/devices {"name": "Desk Lamp", "protocol": "mqtt", "address": "desk_lamp"}
# 3. For Tasmota devices, add config: {"device_style": "tasmota"}
```

## Separate Machine Deployment

Run the server on a powerful machine and the client on any laptop with a mic:

```bash
# Server machine (has GPU for vision, or just CPU)
python run_server.py

# Client machine (needs GPU for Whisper STT)
# Edit .env: SERVER_WS_URL=ws://SERVER_IP:8000/ws/client
python run_client.py
```

## Project Structure

```
├── run_server.py / run_client.py    # Entry points
├── setup.sh / setup.bat             # Automated setup
├── .env.example                     # Environment template
├── server/
│   └── app/
│       ├── main.py                  # FastAPI + WebSocket endpoints
│       ├── orchestrator.py          # Brain: events → LLM → tools → response
│       ├── config.py                # Settings from .env
│       ├── agents/router.py         # Intent classification + domain routing
│       ├── llm/                     # LLM providers (Groq, local SmolVLM)
│       ├── memory/                  # SQLite + FAISS vector memory
│       ├── tools/                   # 60+ tools (email, slack, home, weather, etc.)
│       ├── devices/                 # IoT drivers (HTTP, MQTT, Home Assistant)
│       ├── policy/                  # Confirmation gates, dynamic permissions
│       ├── nlu/                     # JointBERT intent/slot classification
│       ├── vision/                  # Camera, scene analysis, always-on context
│       └── behaviors/               # Proactive behavior rules
├── client/
│   ├── client_main.py               # Voice state machine
│   └── audio/                       # STT, TTS, wake word, speaker verify, VAD
├── shared/schemas/                  # Event and data schemas
├── personality/nova.yaml            # Personality configuration
├── scripts/
│   ├── download_models.py           # Model downloader
│   └── train_jointbert.py           # NLU model training
└── ui/kiosk/                        # Browser-based dashboard
```

## Features

- **Voice Control**: Wake word → local Whisper STT → LLM → streaming TTS (~1.5s latency)
- **FastPath**: JointBERT NLU bypasses LLM for common commands (~50ms end-to-end)
- **Memory**: Structured (people, facts, preferences) + semantic vector search (FAISS)
- **Email**: Gmail + Outlook with OAuth, draft-first workflow
- **Calendar**: Google Calendar integration
- **Slack**: Channel browsing + message posting with confirmation
- **Smart Home**: HTTP, MQTT (Tasmota), and Home Assistant device control
- **Vision**: Always-on camera with SmolVLM scene description
- **Proactive Behaviors**: Welcome-back, stretch reminders, timer alerts
- **60+ Tools**: Productivity, weather, academic, Spotify, and more
- **Speaker Verification**: SpeechBrain ECAPA-TDNN gates sensitive actions
- **Dynamic Permissions**: Voice-configurable confirmation rules ("ask me before sending emails")

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws/client` | WebSocket | Client voice connection |
| `/ws/ui` | WebSocket | UI kiosk connection |
| `/health` | GET | Health check |
| `/api/chat` | POST | Text input (testing) |
| `/api/tools` | GET | List registered tools |
| `/api/devices` | GET/POST | List/register devices |
| `/api/devices/discover-ha` | POST | Auto-discover HA entities |
| `/api/memories` | GET | Browse memories |
| `/api/conversations` | GET | Browse conversation history |
| `/api/video/stream` | GET | MJPEG camera stream |
| `/integrations` | GET | Integration setup UI |
| `/auth/google` | GET | Google OAuth flow |

## Troubleshooting

**Python 3.13 not working**: Use Python 3.11 or 3.12. Python 3.13 has torch/onnxruntime DLL compatibility issues.

**numpy errors**: Ensure `numpy<2` is installed. onnxruntime breaks with numpy 2.x.

**CUDA not detected**: Check `nvidia-smi` works. The system auto-falls back to CPU with a warning.

**Whisper slow on long utterances**: Fixed with sliding window — partials only transcribe the last 10 seconds.

**LLM refuses to send emails**: The system prompt has been hardened against over-cautious safety refusals from Llama models.

**Models not downloading**: Run `python scripts/download_models.py` manually, or models will auto-download on first use.
