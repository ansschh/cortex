"""Conversation orchestrator — the brain that processes events and drives the assistant loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import yaml

from server.app.agents.router import IntentRouter, INTENT_CONVERSATION
from server.app.config import settings
from server.app.llm.base import LLMMessage, LLMResponse
from server.app.llm.groq_provider import GroqProvider
from server.app.memory.context_retriever import ContextRetriever
from server.app.memory.embeddings import EmbeddingService
from server.app.memory.store import MemoryStore
from server.app.memory.vector_index import VectorIndex
from server.app.policy.confirmations import ConfirmationManager
from server.app.policy.gates import DynamicPermissions, PolicyGate, parse_permission_command
from server.app.tools.base import ToolRegistry
from server.app.tools.email_gmail import (
    GmailDraftReplyTool,
    GmailListTool,
    GmailReadTool,
    GmailSendTool,
)
from server.app.tools.email_outlook import OutlookListTool, OutlookReadTool, OutlookSendTool
from server.app.devices.controller import DeviceController
from server.app.devices.registry import DeviceRegistry
from server.app.tools.home import (
    HomeCommandTool,
    HomeDiscoverHATool,
    HomeGetDeviceStateTool,
    HomeListDevicesTool,
    HomeRegisterDeviceTool,
)
from server.app.tools.memory_tools import (
    FactAddTool,
    MemoryAddTool,
    MemorySearchTool,
    PersonAddTool,
    PreferenceSetTool,
)
from server.app.tools.slack import SlackDraftMessageTool, SlackListChannelsTool, SlackPostMessageTool
from server.app.tools.vision import VisionDescribeTool, VisionSnapshotTool
from server.app.tools.productivity import (
    TodoAddTool, TodoListTool, TodoCompleteTool, TodoDeleteTool,
    TimerSetTool, TimerListTool, TimerCancelTool,
    AlarmSetTool, AlarmListTool, AlarmCancelTool,
    ReminderSetTool, ReminderListTool,
)
from server.app.tools.weather import (
    WeatherCurrentTool, WeatherForecastTool, WeatherHourlyTool,
    WeatherAlertsTool, WeatherOutfitTool,
)
from server.app.tools.academic import (
    StudyStartTool, StudyEndTool, StudyStatsTool,
    FlashcardCreateTool, FlashcardQuizTool, FlashcardListDecksTool,
    CalcMathTool, CalcConvertTool,
)
from server.app.tools.spotify import (
    SpotifyPlayTool, SpotifyPauseTool, SpotifySkipTool, SpotifyQueueTool,
    SpotifyNowPlayingTool, SpotifySearchTool, SpotifyVolumeTool, SpotifyPlaylistTool,
)
from server.app.tools.calendar_tools import (
    CalendarListEventsTool, CalendarCreateEventTool, CalendarDeleteEventTool,
    CalendarTodayTool, CalendarTomorrowTool,
)
from server.app.fastpath import FastPath
from server.app.ui.events import ConnectionManager
from shared.schemas.events import (
    AssistantText,
    AssistantTTSText,
    UICard,
    UICardsUpdate,
    UIStatusUpdate,
    UIToast,
)
from shared.schemas.memory import MemoryEntry
from shared.schemas.tool_calls import PendingAction, ToolResult

logger = logging.getLogger(__name__)

_PERSONALITY_PATH = Path(__file__).resolve().parents[2] / "personality" / "nova.yaml"


class Orchestrator:
    """Central orchestrator — receives events, drives LLM + tools, emits responses."""

    def __init__(self, connection_manager: ConnectionManager):
        self.cm = connection_manager
        self.memory = MemoryStore()
        self.llm = GroqProvider()
        self.policy = PolicyGate(settings.speaker_verify_threshold)
        self.dynamic_perms: Optional[DynamicPermissions] = None  # Initialized after DB connects
        self.confirmations = ConfirmationManager()
        self.tools = ToolRegistry()
        self.router = IntentRouter(self.llm)

        # Device control
        self.device_registry: Optional[DeviceRegistry] = None  # Initialized after DB connects
        self.device_controller = DeviceController()

        # Vision
        self.vision_llm = GroqProvider(model=settings.groq_vision_model)
        self.local_vision_llm = None  # Loaded in initialize() for always-on context
        self.vision_context = None  # Initialized in _register_tools after camera is created

        # Proactive behavior system
        self.event_bus = None
        self.behavior_manager = None

        # Infinite memory — embedding + vector search
        self.embeddings = EmbeddingService()
        self.vector_index = VectorIndex(index_path="data/vector_index.bin")
        self.context_retriever: Optional[ContextRetriever] = None

        # Periodic tasks
        self._summarization_task: Optional[asyncio.Task] = None

        # Conversation history (per-session, reset on restart)
        self._history: list[LLMMessage] = []
        self._max_history = 40

        # Session tracking for conversation persistence
        self._session_id = str(uuid.uuid4())[:12]
        self._turn_counter = 0

        # Speaker state
        self._speaker_verified = False
        self._speaker_confidence = 0.0
        self._speaker_label = ""

        # Interruption tracking
        self._interruption_count = 0
        self._was_interrupted = False

        # Speculative FastPath cache (populated by stt_partial, consumed by stt_final)
        self._speculative_cache: Optional[dict] = None

        # Last classified intent (for follow-up context in routing)
        self._last_intent: Optional[str] = None

        # Chat log persistence
        self._chat_log_path = Path(os.getenv("CHAT_LOG_PATH", "data/chat_log.jsonl"))
        self._chat_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Active UI cards
        self._active_cards: list[UICard] = []

        # Load personality
        self._system_prompt = self._load_personality()

    def _load_personality(self) -> str:
        try:
            with open(_PERSONALITY_PATH) as f:
                config = yaml.safe_load(f)
            return config.get("system_prompt", "You are a helpful assistant.")
        except Exception:
            return "You are NOVA, a helpful dorm room AI assistant."

    async def initialize(self) -> None:
        """Initialize memory store, embedding model, vector index, devices, and tools."""
        await self.memory.initialize()

        # Initialize device registry (shares the same DB connection)
        self.device_registry = DeviceRegistry(self.memory._db)
        await self.device_registry.initialize()

        # Initialize dynamic permissions (voice-configurable confirmation rules)
        self.dynamic_perms = DynamicPermissions(self.memory)
        await self.dynamic_perms.load()

        # Initialize integration tables (todos, timers, alarms, etc.)
        from server.app.integrations.schemas import initialize_integration_tables
        await initialize_integration_tables(self.memory._db)

        # Initialize embedding model + vector index
        self.embeddings.initialize()
        self.vector_index.initialize()

        # Wire into memory store for index-on-write
        self.memory.set_embedding_service(self.embeddings)
        self.memory.set_vector_index(self.vector_index)

        # Rebuild vector index if empty but DB has data
        if self.vector_index.is_available and self.vector_index.count == 0:
            rebuilt = await self.memory.rebuild_vector_index(self.embeddings, self.vector_index)
            if rebuilt > 0:
                logger.info(f"Rebuilt vector index with {rebuilt} items from DB")

        # Load local vision model for always-on context (zero API cost)
        if settings.vision_context_enabled and settings.vision_context_local:
            from server.app.llm.local_vision import LocalVisionProvider
            self.local_vision_llm = LocalVisionProvider()
            await self.local_vision_llm.load()

        # Create context retriever
        if self.embeddings.is_available and self.vector_index.is_available:
            self.context_retriever = ContextRetriever(
                self.embeddings, self.vector_index, self.memory,
            )
            logger.info("Context retriever ready — infinite memory active")
        else:
            logger.warning("Embeddings or vector index unavailable — running without memory context")

        # Initialize proactive behavior system (EventBus + BehaviorManager)
        from server.app.behaviors.events import EventBus
        from server.app.behaviors.manager import BehaviorManager
        self.event_bus = EventBus()
        self.behavior_manager = BehaviorManager(
            event_bus=self.event_bus,
            speak_callback=self._proactive_speak,
        )

        self._register_tools()

        # FastPath — JointBERT NLU for sub-50ms tool execution (bypasses all LLM calls)
        self.fastpath = FastPath(self.tools, model_dir="data/models/jointbert")

        # Start behavior manager after tools (vision context publishes to event_bus)
        await self.behavior_manager.start()

        # Start periodic summarization (runs first after 10s, then every 30 min)
        self._summarization_task = asyncio.create_task(self._periodic_summarization())

        logger.info("Orchestrator initialized.")

    def _register_tools(self) -> None:
        # Memory tools
        self.tools.register(MemoryAddTool(self.memory))
        self.tools.register(MemorySearchTool(self.memory))
        self.tools.register(PersonAddTool(self.memory))
        self.tools.register(FactAddTool(self.memory))
        self.tools.register(PreferenceSetTool(self.memory))

        # Email tools
        self.tools.register(GmailListTool())
        self.tools.register(GmailReadTool())
        self.tools.register(GmailDraftReplyTool())
        self.tools.register(GmailSendTool())
        self.tools.register(OutlookListTool())
        self.tools.register(OutlookReadTool())
        self.tools.register(OutlookSendTool())

        # Slack tools
        self.tools.register(SlackDraftMessageTool())
        self.tools.register(SlackPostMessageTool())
        self.tools.register(SlackListChannelsTool())

        # Home control (real device registry + controller)
        self.tools.register(HomeCommandTool(self.device_registry, self.device_controller))
        self.tools.register(HomeListDevicesTool(self.device_registry))
        self.tools.register(HomeRegisterDeviceTool(self.device_registry))
        self.tools.register(HomeGetDeviceStateTool(self.device_registry, self.device_controller))
        self.tools.register(HomeDiscoverHATool(self.device_registry, self.device_controller))

        # Productivity tools (todo, timer, alarm, reminder)
        db = self.memory._db
        self.tools.register(TodoAddTool(db))
        self.tools.register(TodoListTool(db))
        self.tools.register(TodoCompleteTool(db))
        self.tools.register(TodoDeleteTool(db))
        self.tools.register(TimerSetTool(db, event_bus=self.event_bus))
        self.tools.register(TimerListTool(db))
        self.tools.register(TimerCancelTool(db))
        self.tools.register(AlarmSetTool(db))
        self.tools.register(AlarmListTool(db))
        self.tools.register(AlarmCancelTool(db))
        self.tools.register(ReminderSetTool(db, event_bus=self.event_bus))
        self.tools.register(ReminderListTool(db))

        # Weather tools
        owm_key = settings.openweathermap_api_key
        self.tools.register(WeatherCurrentTool(owm_key))
        self.tools.register(WeatherForecastTool(owm_key))
        self.tools.register(WeatherHourlyTool(owm_key))
        self.tools.register(WeatherAlertsTool())
        self.tools.register(WeatherOutfitTool(owm_key))

        # Academic tools (study, flashcard, calc)
        self.tools.register(StudyStartTool(db))
        self.tools.register(StudyEndTool(db))
        self.tools.register(StudyStatsTool(db))
        self.tools.register(FlashcardCreateTool(db))
        self.tools.register(FlashcardQuizTool(db))
        self.tools.register(FlashcardListDecksTool(db))
        self.tools.register(CalcMathTool())
        self.tools.register(CalcConvertTool())

        # Spotify tools
        spotify_token = settings.spotify_access_token
        self.tools.register(SpotifyPlayTool(spotify_token))
        self.tools.register(SpotifyPauseTool(spotify_token))
        self.tools.register(SpotifySkipTool(spotify_token))
        self.tools.register(SpotifyQueueTool(spotify_token))
        self.tools.register(SpotifyNowPlayingTool(spotify_token))
        self.tools.register(SpotifySearchTool(spotify_token))
        self.tools.register(SpotifyVolumeTool(spotify_token))
        self.tools.register(SpotifyPlaylistTool(spotify_token))

        # Google Calendar tools
        self.tools.register(CalendarListEventsTool())
        self.tools.register(CalendarCreateEventTool())
        self.tools.register(CalendarDeleteEventTool())
        self.tools.register(CalendarTodayTool())
        self.tools.register(CalendarTomorrowTool())

        # Vision (real camera + vision LLM + always-on context)
        from server.app.vision.camera import CameraManager
        from server.app.vision.context import VisionContext
        self.camera_manager = CameraManager()
        self.tools.register(VisionSnapshotTool(self.camera_manager))
        self.tools.register(VisionDescribeTool(self.camera_manager, self.vision_llm))

        # Always-on vision context — uses local SmolVLM (zero API cost) or falls back to Groq
        if settings.vision_context_enabled:
            context_llm = self.local_vision_llm if self.local_vision_llm and self.local_vision_llm.is_loaded else self.vision_llm
            self.vision_context = VisionContext(
                camera_manager=self.camera_manager,
                vision_llm=context_llm,
                capture_interval=settings.vision_context_capture_interval,
                change_check_interval=settings.vision_context_change_check_interval,
                describe_interval=settings.vision_context_describe_interval,
                buffer_seconds=settings.vision_context_buffer_seconds,
                max_buffer_size=20,
                skip_static=settings.vision_context_skip_static,
                event_bus=self.event_bus,
            )
            self.vision_context.start()

    async def _proactive_speak(self, prompt: str, max_tokens: int = 80) -> None:
        """Generate and speak a proactive message (called by BehaviorManager).

        The LLM generates natural speech based on the prompt.
        Only speaks if no active conversation is happening.
        """
        try:
            # Build a minimal context for the proactive message
            vision_ctx = self._get_vision_context()
            system = (
                f"You are {settings.assistant_name}, a proactive dorm room AI assistant. "
                "You are about to speak unprompted to the user based on something you noticed. "
                "Be casual, brief, and natural — like a roommate, not a robot. "
                "Keep your response to ONE short sentence.\n"
            )
            if vision_ctx:
                system += f"\n{vision_ctx}\n"

            messages = [
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=prompt),
            ]

            response = await self.llm.chat(messages, temperature=0.8, max_tokens=max_tokens)
            if response.text and response.text.strip():
                text = response.text.strip()
                await self._send_response(text, text)
                # Persist as assistant-initiated turn
                await self._persist_turn("assistant", f"[proactive] {text}", "proactive")
                logger.info(f"Proactive speech: {text[:80]}...")
        except Exception as e:
            logger.error(f"Proactive speak error: {e}")

    async def shutdown(self) -> None:
        # Stop behavior manager
        if self.behavior_manager:
            await self.behavior_manager.stop()

        # Stop vision context
        if self.vision_context:
            self.vision_context.stop()

        # Clean disconnect IoT drivers
        if self.device_controller:
            await self.device_controller.shutdown()

        # Cancel periodic summarization
        if self._summarization_task and not self._summarization_task.done():
            self._summarization_task.cancel()
            try:
                await self._summarization_task
            except asyncio.CancelledError:
                pass

        if self.vector_index.is_available:
            self.vector_index.save()
            logger.info("Vector index saved on shutdown")
        await self.memory.close()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def handle_wakeword(self, data: dict) -> None:
        """Handle wake word detection — update UI status."""
        model_name = data.get("model_name", "")
        if model_name == "barge_in":
            self._was_interrupted = True
            self._interruption_count += 1
            logger.info(f"Barge-in detected (interruption #{self._interruption_count})")
        else:
            self._was_interrupted = False
        await self._update_status("listening")
        await self._send_toast("Listening...", "info")

    async def handle_client_state(self, data: dict) -> None:
        """Forward client state changes to the UI."""
        state = data.get("state", "idle")
        # Map client states to UI-friendly names
        if state == "deep_idle":
            state = "idle"
        elif state == "hot_listening":
            state = "listening"
        await self._update_status(state)

    async def handle_speaker_verified(self, data: dict) -> None:
        self._speaker_verified = data.get("is_verified", False)
        self._speaker_confidence = data.get("confidence", 0.0)
        self._speaker_label = data.get("speaker_label", "")
        await self._update_status("listening")

    async def handle_stt_partial(self, data: dict) -> None:
        text = data.get("text", "")
        await self.cm.send_to_ui(UIStatusUpdate(
            assistant_state="listening",
            speaker_verified=self._speaker_verified,
            speaker_label=self._speaker_label,
            transcript=text,
        ).model_dump())

        # --- SPECULATIVE FASTPATH: pre-run NLU on partial transcripts ---
        if text.strip() and len(text.strip()) >= 5:
            try:
                fp_result = self.fastpath.try_match(text.strip())
                if fp_result and fp_result.confidence >= 0.85:
                    self._speculative_cache = {
                        "text": text.strip(),
                        "intent": fp_result.intent,
                        "result": fp_result,
                    }
                    logger.debug(f"Speculative FP: {fp_result.intent} ({fp_result.confidence:.2f}) on partial '{text[:50]}'")
            except Exception:
                pass  # Never let speculative work break the pipeline

    @staticmethod
    def _normalize_spoken_text(text: str) -> str:
        """Normalize speech artifacts — spoken email addresses, spelled-out words, etc."""
        # "at the rate" / "at the rate of" → "@"
        text = re.sub(r'\bat the rate(?:\s+of)?\b', '@', text, flags=re.IGNORECASE)
        # "dot com", "dot edu", "dot org" etc → ".com" etc (only after @ or letter)
        text = re.sub(r'\bdot\s+(com|edu|org|net|io|gov|co|ai)\b', r'.\1', text, flags=re.IGNORECASE)
        # General "dot" in email context (between @ and end, or between words that look like domain parts)
        # Only apply if there's an @ nearby
        if '@' in text:
            at_pos = text.index('@')
            domain_part = text[at_pos:]
            domain_part = re.sub(r'\s+dot\s+', '.', domain_part, flags=re.IGNORECASE)
            text = text[:at_pos] + domain_part
        # Collapse spaces around @ sign
        text = re.sub(r'\s*@\s*', '@', text)
        # Detect spelled-out letters: "D H A S H M I" → "dhashmi"
        # Match 3+ single uppercase letters separated by spaces
        def collapse_spelled(m):
            return m.group(0).replace(' ', '').lower()
        text = re.sub(r'\b([A-Z](?:\s+[A-Z]){2,})\b', collapse_spelled, text)
        return text

    async def handle_stt_final(self, data: dict) -> None:
        """Main entry point: user said something → process it."""
        text = data.get("text", "").strip()
        if not text:
            return

        # Normalize spoken artifacts (email addresses, spelled-out words)
        text = self._normalize_spoken_text(text)

        request_id = str(uuid.uuid4())[:8]
        await self._update_status("thinking")
        asyncio.create_task(self.memory.log_audit("stt_final", {"text": text, "request_id": request_id}))

        # --- FASTPATH: bypass ALL LLM calls for ML-classified requests ---
        # Check speculative cache first (populated during stt_partial)
        fp_result = None
        speculative_hit = False
        cached = self._speculative_cache
        self._speculative_cache = None  # always consume

        if cached and cached.get("intent"):
            # Re-run NLU on final text but reuse cache if intent matches
            fp_result = self.fastpath.try_match(text)
            if fp_result and fp_result.intent == cached["intent"]:
                speculative_hit = True
                logger.info(f"Speculative FP HIT: {fp_result.intent} (partial matched final)")
        if not fp_result:
            fp_result = self.fastpath.try_match(text)

        if fp_result:
            t0 = time.perf_counter()
            tool_result, _template = await self.fastpath.execute(fp_result)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            spec_tag = " [speculative]" if speculative_hit else ""
            logger.info(f"FastPath [{fp_result.tool_name}] intent={fp_result.intent} conf={fp_result.confidence:.2f} tool={elapsed_ms:.1f}ms{spec_tag}")

            # Stream the response through LLM for natural voice
            self._history.append(LLMMessage(role="user", content=text))
            self._trim_history()
            asyncio.create_task(self._persist_turn("user", text, "fastpath"))

            await self._stream_fastpath_response(text, fp_result, tool_result, request_id)
            await self._update_status("idle")
            return

        # Check for permission-setting voice commands ("ask me before sending emails")
        perm_cmd = parse_permission_command(text)
        if perm_cmd:
            await self._handle_permission_command(perm_cmd)
            await self._update_status("idle")
            return

        # Check for deterministic memory command
        if self.policy.is_memory_command(text):
            await self._handle_memory_command(text, request_id)
            return

        # Check for confirmation phrase (for pending actions)
        if self.policy.is_confirmation_phrase(text):
            await self._handle_confirmation(request_id)
            return

        # Normal LLM conversation
        await self._handle_conversation(text, request_id)

    async def handle_user_confirmation(self, data: dict) -> None:
        """Handle explicit confirmation from UI button or voice."""
        confirmed = data.get("confirmed", False)
        action_id = data.get("pending_action_id", "")

        if action_id:
            action = self.confirmations.resolve(action_id, confirmed)
        else:
            action = self.confirmations.resolve_latest(confirmed)

        if not action:
            await self._send_response("No pending action to confirm.", "Nothing to confirm.")
            return

        if confirmed:
            await self._execute_confirmed_action(action)
        else:
            await self._send_response("Action cancelled.", "Cancelled.")
            await self._send_toast("Action cancelled.", "warning")

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def _handle_memory_command(self, text: str, request_id: str) -> None:
        memory_text = self.policy.extract_memory_text(text)
        tool = self.tools.get("memory.add")
        result = await tool.execute(text=memory_text)

        response = f"Got it. I've saved that to memory: \"{memory_text}\""
        await self._send_response(response, response)
        if result.display_card:
            await self._add_card(UICard(**result.display_card))
        await self.memory.log_audit("memory_add", {"text": memory_text, "request_id": request_id})

    async def _handle_confirmation(self, request_id: str) -> None:
        action = self.confirmations.resolve_latest(True)
        if not action:
            await self._send_response("There's nothing pending to confirm.", "Nothing to confirm.")
            return

        if action.requires_speaker_verification and not self._speaker_verified:
            self.confirmations.create_pending(
                action.tool_name, action.arguments, action.preview_text,
                action.requires_speaker_verification,
            )
            await self._send_response(
                "I need to verify it's you before doing that. Please speak again for verification.",
                "Speaker verification required.",
            )
            return

        await self._execute_confirmed_action(action)

    async def _handle_permission_command(self, cmd: dict) -> None:
        """Handle voice-configurable permission commands."""
        if not self.dynamic_perms:
            await self._send_response("Permission system isn't ready yet.", "Permission system not ready.")
            return

        action = cmd.get("action")
        if action == "list":
            rules = self.dynamic_perms.list_rules()
            if not rules:
                await self._send_response(
                    "No custom confirmation rules set. High-risk actions like sending emails "
                    "and posting to Slack require confirmation by default.",
                    "No custom confirmation rules."
                )
            else:
                lines = []
                for tool, requires in rules.items():
                    status = "requires confirmation" if requires else "no confirmation needed"
                    lines.append(f"  {tool}: {status}")
                text = "Here are your confirmation rules:\n" + "\n".join(lines)
                await self._send_response(text, text)
            return

        tool_pattern = cmd.get("tool_pattern", "")
        desc = cmd.get("description", tool_pattern)

        if action == "enable":
            await self.dynamic_perms.set_rule(tool_pattern, True)
            await self._send_response(
                f"Got it, I'll ask you before {desc} from now on.",
                f"Got it, I'll ask you before {desc} from now on."
            )
        elif action == "disable":
            await self.dynamic_perms.set_rule(tool_pattern, False)
            await self._send_response(
                f"Alright, I won't ask before {desc} anymore.",
                f"Alright, I won't ask before {desc} anymore."
            )

    async def _execute_confirmed_action(self, action: PendingAction) -> None:
        tool = self.tools.get(action.tool_name)
        if not tool:
            await self._send_response(f"Tool {action.tool_name} not found.", "Error: tool not found.")
            return

        await self._update_status("thinking")
        result = await tool.safe_execute(**action.arguments)
        await self.memory.log_audit("tool_executed", {
            "tool": action.tool_name, "args": action.arguments,
            "success": result.success, "confirmed": True,
        })

        if result.success:
            await self._send_response(f"Done. {action.tool_name} executed successfully.", "Done!")
            if result.display_card:
                await self._add_card(UICard(**result.display_card))
        else:
            await self._send_response(f"Failed: {result.error}", f"Error: {result.error}")

        await self._update_status("idle")

    async def _build_dynamic_prompt(self, user_text: str, agent_spec) -> str:
        """Build a system prompt with personality + retrieved memory context + vision context."""
        base_prompt = agent_spec.system_prompt
        if base_prompt == "__PERSONALITY__":
            base_prompt = self._system_prompt

        context_section = None
        if self.context_retriever:
            try:
                bundle = await self.context_retriever.retrieve(user_text)
                context_section = bundle.format_for_prompt()
            except Exception as e:
                logger.error(f"Context retrieval failed: {e}")

        return self._build_prompt_with_context(agent_spec, context_section)

    # Trivial queries that don't need memory retrieval
    _TRIVIAL_PATTERNS = re.compile(
        r"^(hey|hi|hello|yo|sup|thanks|thank you|ok|okay|yeah|yep|yup|nah|no|nope|"
        r"bye|goodbye|good morning|good night|what's up|how are you|hm+|huh|sure|"
        r"got it|cool|nice|great|alright|sounds good|never mind)[\.\!\?]?$",
        re.IGNORECASE,
    )

    async def _retrieve_context_if_needed(self, text: str) -> str | None:
        """Retrieve memory context for the user's query. Skip for trivial/short queries."""
        if not self.context_retriever:
            return None

        # Skip for trivial greetings / single-word responses
        if self._TRIVIAL_PATTERNS.match(text.strip()):
            logger.info("Skipping memory retrieval for trivial query")
            return None

        try:
            bundle = await self.context_retriever.retrieve(text)
            return bundle.format_for_prompt()
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return None

    def _build_prompt_with_context(self, agent_spec, context_section: str | None) -> str:
        """Build system prompt with personality + pre-fetched memory context + vision context."""
        base_prompt = agent_spec.system_prompt
        if base_prompt == "__PERSONALITY__":
            base_prompt = self._system_prompt

        parts = [base_prompt]

        if context_section:
            parts.append(
                "--- PERSONAL CONTEXT ---\n"
                "Use this information naturally — don't say 'according to my memory', "
                "just know it like a real person would.\n\n"
                f"{context_section}"
            )

        # CRITICAL: Force tool usage — Llama models tend to role-play instead of calling tools
        parts.append(
            "--- TOOL USAGE (CRITICAL) ---\n"
            "You have function-calling tools available. You MUST use them for any action that has a matching tool.\n"
            "NEVER pretend to perform an action by just saying you did it. If the user asks to create a calendar event, "
            "you MUST call the calendar_create_event tool. If they ask to send an email, MUST call the email tool.\n"
            "If they ask for weather, MUST call the weather tool. If they ask to set a timer, MUST call the timer tool.\n"
            "DO NOT fake or role-play tool actions. The tools are real and connected to real services.\n"
            "If a tool is available for what the user wants, ALWAYS call it. Never skip it."
        )

        # Tool error handling guidance
        parts.append(
            "--- TOOL ERROR HANDLING ---\n"
            "When a tool returns an error:\n"
            "- [MISSING_API_KEY]: Tell the user casually, like \"yo, that integration isn't hooked up yet — "
            "you'll need to add the API key. Hit up localhost:8000/integrations to set it up.\"\n"
            "- [INVALID_API_KEY]: Say something like \"hmm, looks like your key for X is expired or broken. "
            "Might wanna grab a new one from their dashboard.\"\n"
            "- [RATE_LIMITED]: \"We're getting throttled by X's API — give it a sec and try again.\"\n"
            "- [NETWORK_ERROR]: \"Can't reach X right now — might be a connection issue on our end.\"\n"
            "- [TOOL_ERROR]: Explain what broke naturally. If it's a config issue, mention the setup page.\n"
            "Never show raw error messages or tracebacks. Make it sound like YOU noticed the problem, "
            "like a roommate saying \"oh yeah that thing's broken, here's what you gotta do.\""
        )

        # Inject always-on vision context (available for ALL intents)
        vision_ctx = self._get_vision_context()
        if vision_ctx:
            parts.append(
                "--- VISUAL AWARENESS ---\n"
                "You have a camera that sees the room. Below is what you've observed recently. "
                "Use this naturally when relevant — you can reference what the user is doing, "
                "wearing, holding, or showing without them asking. "
                "If they ask about something visual, use this context to answer immediately.\n\n"
                f"{vision_ctx}"
            )

        return "\n\n".join(parts)

    def _get_vision_context(self) -> str:
        """Get recent visual observations from the always-on vision context."""
        if not self.vision_context or not self.vision_context.is_running:
            return ""
        return self.vision_context.get_recent_context(seconds=30.0, max_entries=8)

    @staticmethod
    def _detect_follow_up_expected(response_text: str) -> bool:
        """Check if the assistant's response expects a user reply (ends with question, etc)."""
        text = response_text.strip()
        if not text:
            return False
        # Ends with question mark
        if text[-1] == "?":
            return True
        # Common question patterns in last sentence
        last_sentence = text.split(".")[-1].strip().lower()
        question_phrases = [
            "what do you think", "want me to", "should i", "would you like",
            "does that work", "sound good", "let me know", "anything else",
            "what about", "how about", "shall i", "do you want",
        ]
        return any(phrase in last_sentence for phrase in question_phrases)

    async def _handle_conversation(self, text: str, request_id: str) -> None:
        """Route user message to the right agent, then stream response sentence-by-sentence."""
        _conv_t0 = time.perf_counter()

        # Step 1: Classify intent + retrieve memory context IN PARALLEL
        intent_task = asyncio.create_task(self.router.classify(text, prev_intent=self._last_intent))
        context_task = asyncio.create_task(self._retrieve_context_if_needed(text))
        intent, context_section = await asyncio.gather(intent_task, context_task)

        # Track the last intent for follow-up context
        self._last_intent = intent

        agent_spec = self.router.get_agent_spec(intent)
        agent_tools = self.router.get_tools_for_intent(intent, self.tools)

        logger.info(f"Intent: {intent} | tools: {len(agent_tools)} | request: {request_id}")

        # Step 2: Add user message
        user_content = text
        if self._was_interrupted and self._interruption_count >= 3:
            user_content = f"[USER INTERRUPTED] {text}"
        self._was_interrupted = False

        self._history.append(LLMMessage(role="user", content=user_content))
        self._trim_history()

        # Persist user turn in background (don't block response)
        asyncio.create_task(self._persist_turn("user", user_content, intent))

        # Build dynamic prompt using pre-fetched context
        dynamic_prompt = self._build_prompt_with_context(agent_spec, context_section)
        messages = [LLMMessage(role="system", content=dynamic_prompt)] + self._history

        # Step 3: If tools needed, use non-streaming chat
        if agent_tools:
            logger.info(f"Sending {len(agent_tools)} tools to LLM: {[t['function']['name'] for t in agent_tools]}")
            try:
                response = await self.llm.chat(
                    messages=messages,
                    tools=agent_tools,
                    temperature=0.8,
                    max_tokens=2048,
                )
            except Exception as e:
                import traceback
                logger.error(f"LLM error: {e}\n{traceback.format_exc()}")
                await self._send_response("Sorry, I had trouble thinking about that. Try again?", "LLM error.")
                await self._update_status("idle")
                return

            logger.info(f"LLM response: tool_calls={len(response.tool_calls)}, text={response.text[:200] if response.text else '(empty)'}, finish={response.finish_reason}")
            if response.tool_calls:
                await self._process_tool_calls(response, request_id, agent_spec)
            else:
                self._history.append(LLMMessage(role="assistant", content=response.text))
                await self._persist_turn("assistant", response.text, intent)
                await self._send_response(response.text, response.text)

            self._log_chat(text, response.text, intent, request_id)
            await self._update_status("idle")
            return

        # Step 4: Stream LLM response clause-by-clause for minimum-latency TTS
        # Signal client to prepare BEFORE LLM starts (client opens audio output + uses preconnected TTS)
        await self.cm.send_to_client({"event": "assistant_tts_start"})

        full_response = ""
        sentence_buffer = ""
        sentence_count = 0

        try:
            async for token in self.llm.chat_stream(
                messages=messages,
                temperature=0.8,
                max_tokens=2048,
            ):
                full_response += token
                sentence_buffer += token

                # Check if we have a complete sentence
                if self._is_sentence_end(sentence_buffer):
                    sentence = sentence_buffer.strip()
                    if sentence:
                        sentence_count += 1
                        logger.info(f"Streaming sentence #{sentence_count}: '{sentence[:60]}...'")
                        await self.cm.send_to_client({
                            "event": "assistant_tts_chunk",
                            "text": sentence,
                        })
                    sentence_buffer = ""

            # Send any remaining text
            remaining = sentence_buffer.strip()
            if remaining:
                sentence_count += 1
                logger.info(f"Streaming sentence #{sentence_count} (final): '{remaining[:60]}...'")
                await self.cm.send_to_client({
                    "event": "assistant_tts_chunk",
                    "text": remaining,
                })

            # Signal end of streaming with follow-up flag
            follow_up = self._detect_follow_up_expected(full_response)
            await self.cm.send_to_client({
                "event": "assistant_tts_end",
                "follow_up_expected": follow_up,
            })

        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            # Fallback: if streaming failed and we have partial text, send it
            if full_response.strip():
                await self.cm.send_to_client({
                    "event": "assistant_tts_chunk",
                    "text": full_response.strip(),
                })
                await self.cm.send_to_client({"event": "assistant_tts_end"})
            else:
                await self._send_response(
                    "Sorry, I had trouble thinking about that. Try again?", "LLM error."
                )
                await self._update_status("idle")
                return

        # Update history, UI, and log (persist in background)
        self._history.append(LLMMessage(role="assistant", content=full_response))
        asyncio.create_task(self._persist_turn("assistant", full_response, intent))
        await self.cm.send_to_client(AssistantText(text=full_response).model_dump())
        await self._add_card(UICard(
            card_type="AssistantResponseCard",
            title=settings.assistant_name,
            body=full_response,
            priority=4,
        ))
        self._log_chat(text, full_response, intent, request_id)
        _conv_elapsed = (time.perf_counter() - _conv_t0) * 1000
        logger.info(f"LLM path total: {_conv_elapsed:.0f}ms (intent={intent}, request={request_id})")
        await self._update_status("idle")

    async def _stream_fastpath_response(
        self, user_text: str, fp_result, tool_result, request_id: str,
    ) -> None:
        """FastPath hybrid: tool already executed, now stream LLM response for natural TTS."""
        # Build a compact prompt — the tool is already done, we just need a natural reply
        tool_summary = str(tool_result.result) if tool_result.result else "Done."
        if tool_result.error:
            tool_summary = f"Error: {tool_result.error}"

        fp_system = (
            f"{self._system_prompt}\n\n"
            "--- CONTEXT ---\n"
            f"You already executed the tool '{fp_result.tool_name}' for the user.\n"
            f"Tool result: {tool_summary}\n\n"
            "Respond to the user naturally and briefly based on this result. "
            "Don't mention tools or technical details — just answer like a helpful roommate. "
            "Keep it short (1-2 sentences max unless the result has a lot of info to relay)."
        )

        messages = [LLMMessage(role="system", content=fp_system)] + self._history

        # Stream clause-by-clause to TTS (same pattern as _handle_conversation)
        await self.cm.send_to_client({"event": "assistant_tts_start"})

        full_response = ""
        sentence_buffer = ""
        sentence_count = 0

        try:
            async for token in self.llm.chat_stream(
                messages=messages, temperature=0.8, max_tokens=512,
            ):
                full_response += token
                sentence_buffer += token

                if self._is_sentence_end(sentence_buffer):
                    sentence = sentence_buffer.strip()
                    if sentence:
                        sentence_count += 1
                        await self.cm.send_to_client({
                            "event": "assistant_tts_chunk",
                            "text": sentence,
                        })
                    sentence_buffer = ""

            remaining = sentence_buffer.strip()
            if remaining:
                sentence_count += 1
                await self.cm.send_to_client({
                    "event": "assistant_tts_chunk",
                    "text": remaining,
                })

            follow_up = self._detect_follow_up_expected(full_response)
            await self.cm.send_to_client({
                "event": "assistant_tts_end",
                "follow_up_expected": follow_up,
            })

        except Exception as e:
            logger.error(f"FastPath LLM streaming error: {e}")
            if full_response.strip():
                await self.cm.send_to_client({
                    "event": "assistant_tts_chunk",
                    "text": full_response.strip(),
                })
                await self.cm.send_to_client({"event": "assistant_tts_end"})
            else:
                # Fallback to a generic acknowledgement
                await self._send_response("Done.", "Done.")
                return

        # Persist assistant response
        self._history.append(LLMMessage(role="assistant", content=full_response))
        asyncio.create_task(self._persist_turn("assistant", full_response, "fastpath"))
        await self.cm.send_to_client(AssistantText(text=full_response).model_dump())
        await self._add_card(UICard(
            card_type="AssistantResponseCard",
            title=settings.assistant_name,
            body=full_response,
            priority=4,
        ))
        self._log_chat(user_text, full_response, "fastpath", request_id)
        logger.info(f"FastPath LLM response: {sentence_count} chunks, {len(full_response)} chars")

    @staticmethod
    def _is_sentence_end(text: str) -> bool:
        """Check if the buffer ends with a speakable clause boundary.

        Aggressive chunking: emit on commas, semicolons, colons, dashes,
        and sentence-ending punctuation. This gets text to TTS faster
        (clause-level instead of sentence-level).
        """
        stripped = text.rstrip()
        if not stripped:
            return False

        last = stripped[-1]

        # Sentence-ending punctuation — always emit
        if last in ".!?":
            if len(stripped) > 8:
                return True

        # Clause-level punctuation — emit if we have enough text to sound natural
        if last in ",;:—–" and len(stripped) > 15:
            return True

        # Long dash mid-sentence
        if stripped.endswith(" -") and len(stripped) > 15:
            return True

        return False

    def _log_chat(self, user_text: str, assistant_text: str, intent: str, request_id: str) -> None:
        """Append a chat exchange to the persistent JSONL log."""
        try:
            entry = {
                "timestamp": time.time(),
                "request_id": request_id,
                "intent": intent,
                "user": user_text,
                "assistant": assistant_text,
                "interrupted": self._was_interrupted,
                "interruption_count": self._interruption_count,
            }
            with open(self._chat_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Chat log write error: {e}")

    async def _process_tool_calls(self, response: LLMResponse, request_id: str, agent_spec=None) -> None:
        """Execute tool calls from the LLM in an agentic loop, respecting policy gates.

        The LLM can chain multiple rounds of tool calls (up to max_agentic_steps).
        Each round: execute tools → get follow-up with tools still available → repeat if more calls.
        Confirmation-gated tools break the loop immediately.
        """
        max_steps = settings.max_agentic_steps
        current_response = response

        # Resolve tools available for follow-ups
        agent_tools = []
        if agent_spec:
            agent_tools = self.router.get_tools_for_intent(agent_spec.intent, self.tools)

        for step in range(max_steps):
            for tc in current_response.tool_calls:
                func = tc.get("function", {})
                logger.info(f"Agentic step {step + 1}/{max_steps} — tool: {func.get('name')} args: {func.get('arguments', '')[:200]}")

            # Add the assistant message with tool calls to history
            self._history.append(LLMMessage(
                role="assistant", content=current_response.text,
                tool_calls=current_response.tool_calls,
            ))

            hit_confirmation = False

            for tc in current_response.tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    arguments = json.loads(func.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}

                tool = self.tools.get(tool_name)
                if not tool:
                    logger.warning(f"Tool '{tool_name}' not found in registry! LLM hallucinated a tool name.")
                    self._history.append(LLMMessage(
                        role="tool",
                        content=f"Error: tool '{tool_name}' does not exist. Available tools: {[t.name for t in self.tools.list_tools()][:20]}. Use an existing tool name.",
                        tool_call_id=tc.get("id", ""),
                    ))
                    continue
                logger.info(f"Executing tool: {tool_name} (canonical: {tool.name})")

                # Policy gate check (with dynamic permission overrides)
                tool_def = tool.get_definition()
                gate_result = self.policy.evaluate(
                    tool_def, self._speaker_verified, self._speaker_confidence,
                    dynamic_perms=self.dynamic_perms,
                )

                if gate_result.requires_confirmation:
                    preview = f"{tool_name}({json.dumps(arguments, indent=2)})"
                    pending = self.confirmations.create_pending(
                        tool_name, arguments, preview,
                        gate_result.requires_speaker_verification,
                    )
                    await self._send_response(
                        f"I've prepared to run {tool_name}. {gate_result.reason} Say 'send it' or 'confirm' to proceed.",
                        f"Pending: {tool_name}",
                    )
                    await self._add_card(UICard(
                        card_type="PendingActionCard",
                        card_id=f"pending-{pending.action_id}",
                        title=f"Pending: {tool_name}",
                        body={"preview": preview, "action_id": pending.action_id},
                        actions=[
                            {"label": "CONFIRM", "action": f"confirm_{pending.action_id}"},
                            {"label": "CANCEL", "action": f"cancel_{pending.action_id}"},
                        ],
                        priority=10,
                    ))
                    self._history.append(LLMMessage(
                        role="tool",
                        content="Action pending confirmation. Awaiting user approval.",
                        tool_call_id=tc.get("id", ""),
                    ))
                    await self.memory.log_audit("tool_pending", {
                        "tool": tool_name, "args": arguments, "request_id": request_id,
                    })
                    hit_confirmation = True
                    continue

                if not gate_result.allowed:
                    self._history.append(LLMMessage(
                        role="tool", content=f"Blocked: {gate_result.reason}",
                        tool_call_id=tc.get("id", ""),
                    ))
                    continue

                # Execute the tool (safe_execute catches errors and returns friendly messages)
                result = await tool.safe_execute(**arguments)
                logger.info(f"Tool {tool_name} result: success={result.success}, error={result.error}, result_preview={str(result.result)[:300]}")

                await self.memory.log_audit("tool_executed", {
                    "tool": tool_name, "args": arguments,
                    "success": result.success, "request_id": request_id,
                })

                self._history.append(LLMMessage(
                    role="tool",
                    content=json.dumps(
                        result.result if result.success else {"error": result.error},
                        default=str,
                    ),
                    tool_call_id=tc.get("id", ""),
                ))

                if result.display_card:
                    await self._add_card(UICard(**result.display_card))

            # If a confirmation was needed, stop the loop — wait for user
            if hit_confirmation:
                return

            # Get follow-up from LLM WITH tools still available (enables chaining)
            followup_prompt = self._system_prompt
            if agent_spec:
                try:
                    followup_prompt = await self._build_dynamic_prompt(
                        self._history[-1].content if self._history else "", agent_spec,
                    )
                except Exception:
                    pass

            messages = [LLMMessage(role="system", content=followup_prompt)] + self._history
            try:
                followup = await self.llm.chat(
                    messages=messages,
                    tools=agent_tools if agent_tools else None,
                    temperature=0.7,
                    max_tokens=1024,
                )
            except Exception as e:
                logger.error(f"LLM follow-up error (step {step + 1}): {e}")
                break

            followup_tool_names = [tc.get("function", {}).get("name", "?") for tc in followup.tool_calls]
            logger.info(f"Follow-up: tool_calls={followup_tool_names}, text={followup.text[:200] if followup.text else '(empty)'}, finish={followup.finish_reason}")

            # If follow-up has more tool calls, continue the loop
            if followup.tool_calls:
                current_response = followup
                continue

            # No more tool calls — emit final text and break
            self._history.append(LLMMessage(role="assistant", content=followup.text))
            await self._persist_turn("assistant", followup.text)
            await self._send_response(followup.text, followup.text)
            return

        # If we exhausted all steps, emit whatever we have
        logger.warning(f"Agentic loop hit max steps ({max_steps})")
        try:
            messages = [LLMMessage(role="system", content=self._system_prompt)] + self._history
            final = await self.llm.chat(messages=messages, temperature=0.7, max_tokens=512)
            self._history.append(LLMMessage(role="assistant", content=final.text))
            await self._persist_turn("assistant", final.text)
            await self._send_response(final.text, final.text)
        except Exception as e:
            logger.error(f"Final follow-up error: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    async def _send_response(self, display_text: str, tts_text: str, use_local_tts: bool = False) -> None:
        await self.cm.send_to_client(AssistantText(text=display_text).model_dump())
        await self.cm.send_to_client(AssistantTTSText(text=tts_text, use_local_tts=use_local_tts).model_dump())
        await self._add_card(UICard(
            card_type="AssistantResponseCard",
            title=settings.assistant_name,
            body=display_text,
            priority=4,
        ))

    async def _send_toast(self, message: str, level: str = "info") -> None:
        await self.cm.send_to_ui(UIToast(message=message, level=level).model_dump())

    async def _update_status(self, state: str) -> None:
        await self.cm.send_to_ui(UIStatusUpdate(
            assistant_state=state,
            speaker_verified=self._speaker_verified,
            speaker_label=self._speaker_label,
        ).model_dump())

    async def _add_card(self, card: UICard) -> None:
        # Replace existing card with same card_id, or append
        if card.card_id:
            self._active_cards = [c for c in self._active_cards if c.card_id != card.card_id]
        self._active_cards.append(card)
        # Keep only recent cards
        if len(self._active_cards) > 20:
            self._active_cards = self._active_cards[-20:]
        await self.cm.send_to_ui(UICardsUpdate(
            cards=[c.model_dump() for c in self._active_cards],
        ).model_dump())

    # ------------------------------------------------------------------
    # Conversation persistence + memory
    # ------------------------------------------------------------------

    async def _persist_turn(self, role: str, content: str, intent: str = "conversation") -> None:
        """Persist a conversation turn to the DB and vector index."""
        try:
            await self.memory.add_conversation_turn(
                self._session_id, role, content, intent,
            )
            self._turn_counter += 1
            # Save vector index to disk every 20 turns
            if self._turn_counter % 20 == 0 and self.vector_index.is_available:
                self.vector_index.save()
        except Exception as e:
            logger.error(f"Failed to persist turn: {e}")

    async def _periodic_summarization(self) -> None:
        """Background task: run summarization after 10s, then every 30 minutes."""
        try:
            await asyncio.sleep(10)
            count = await self.summarize_old_conversations()
            if count > 0:
                logger.info(f"Startup summarization: {count} sessions summarized")
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"Startup summarization error: {e}")

        while True:
            try:
                await asyncio.sleep(1800)  # 30 minutes
                count = await self.summarize_old_conversations()
                if count > 0:
                    logger.info(f"Periodic summarization: {count} sessions summarized")
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Periodic summarization error: {e}")

    async def summarize_old_conversations(self) -> int:
        """Summarize old conversation sessions into episodes for long-term memory.

        Finds sessions >24h old with >4 turns, asks LLM to summarize, stores as episodes.
        Returns number of sessions summarized.
        """
        from shared.schemas.memory import Episode

        try:
            sessions = await self.memory.get_unsummarized_sessions(older_than_hours=24, limit=5)
        except Exception as e:
            logger.error(f"Failed to fetch unsummarized sessions: {e}")
            return 0

        summarized = 0
        for session in sessions:
            transcript = session["transcript"]
            if not transcript:
                continue

            # Truncate very long transcripts
            if len(transcript) > 4000:
                transcript = transcript[:4000] + "\n... (truncated)"

            try:
                messages = [
                    LLMMessage(
                        role="system",
                        content=(
                            "Summarize this conversation in 2-3 concise sentences. "
                            "Focus on key facts, decisions, and personal details mentioned. "
                            "Write in third person ('The user...')."
                        ),
                    ),
                    LLMMessage(role="user", content=transcript),
                ]
                response = await self.llm.chat(messages=messages, temperature=0.3, max_tokens=200)
                await self.memory.add_episode(Episode(summary=response.text))
                summarized += 1
                logger.info(f"Summarized session {session['session_id']} ({session['turn_count']} turns)")
            except Exception as e:
                logger.error(f"Failed to summarize session {session['session_id']}: {e}")

        if summarized > 0 and self.vector_index.is_available:
            self.vector_index.save()
        return summarized
