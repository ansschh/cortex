"""Intent router — classifies user messages and dispatches to the right agent.

Architecture:
  User input → Router (lightweight LLM classify) → domain agent (with domain-specific tools)
  
  conversation  → no tools, pure chat
  email         → Gmail + Outlook tools only
  slack         → Slack tools only
  smart_home    → home control tools only
  memory        → memory tools only
  vision        → camera/vision tools only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from server.app.llm.base import LLMMessage, LLMProvider, LLMResponse
from server.app.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent categories
# ---------------------------------------------------------------------------

INTENT_CONVERSATION = "conversation"
INTENT_EMAIL = "email"
INTENT_SLACK = "slack"
INTENT_SMART_HOME = "smart_home"
INTENT_MEMORY = "memory"
INTENT_VISION = "vision"
INTENT_GENERAL = "general"
INTENT_PRODUCTIVITY = "productivity"
INTENT_WEATHER = "weather"
INTENT_MUSIC = "music"
INTENT_ACADEMIC = "academic"
INTENT_CALENDAR = "calendar"

ALL_INTENTS = [
    INTENT_CONVERSATION,
    INTENT_EMAIL,
    INTENT_SLACK,
    INTENT_SMART_HOME,
    INTENT_MEMORY,
    INTENT_VISION,
    INTENT_GENERAL,
    INTENT_PRODUCTIVITY,
    INTENT_WEATHER,
    INTENT_MUSIC,
    INTENT_ACADEMIC,
    INTENT_CALENDAR,
]

ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a dorm room AI assistant called NOVA.

Given the user's message, classify it into EXACTLY ONE category. Respond with ONLY the category name.

Categories:
- conversation: General chat, greetings, questions about capabilities, opinions, jokes, small talk, help requests, or anything that does NOT require an external action.
- email: Reading, listing, drafting, or sending emails via Gmail or Outlook.
- slack: Sending, drafting, or listing Slack messages or channels.
- smart_home: Controlling devices like lights, fans, locks, speakers, or asking about device status.
- memory: Explicitly asking to remember something, recall a memory, store facts, add people, or set preferences.
- vision: Taking photos, capturing camera snapshots, describing what the camera sees, OR any question that requires LOOKING at the user or their surroundings.
- productivity: To-do lists, timers, alarms, reminders — task management and time management.
- weather: Weather conditions, forecasts, temperature, rain, outfit suggestions based on weather.
- music: Playing music, controlling Spotify, searching for songs, playlists, volume control.
- academic: Study sessions, flashcards, quizzes, math calculations, unit conversions.
- calendar: Calendar events, scheduling, meetings, appointments, today's/tomorrow's schedule, free time.
- general: Requests that span MULTIPLE categories — e.g. "email Bob about the lights" (email + smart_home), "set a reminder to check the weather" (productivity + weather).

IMPORTANT RULES:
- If the previous intent is provided, consider that the user may be continuing that topic.
  Short follow-ups like "yeah", "no", corrections, email addresses, or clarifications likely
  continue the previous domain. Only switch intents if the user clearly changes topic.
- If the user is just asking a question or having a conversation, ALWAYS classify as "conversation".
- Only classify as a tool category if the user EXPLICITLY wants to perform that action.
- If a request clearly involves TWO OR MORE domains, classify as "general".
- "What can you do?" → conversation
- "Tell me a joke" → conversation
- "Set a timer for 5 minutes" → productivity
- "What's the weather?" → weather
- "Play some jazz" → music
- "Add milk to my todo list" → productivity
- "Quiz me on biology" → academic
- "What's 25 times 37?" → academic
- "What's on my calendar today?" → calendar
- "Schedule a meeting at 3pm" → calendar

Respond with ONLY the category name, nothing else."""


# ---------------------------------------------------------------------------
# Agent specs — each domain agent gets its own system prompt + tool subset
# ---------------------------------------------------------------------------

@dataclass
class AgentSpec:
    """Definition of a domain-specific agent."""
    intent: str
    system_prompt: str
    tool_prefixes: list[str] = field(default_factory=list)


AGENT_SPECS: dict[str, AgentSpec] = {
    INTENT_CONVERSATION: AgentSpec(
        intent=INTENT_CONVERSATION,
        system_prompt="__PERSONALITY__",
        tool_prefixes=[],
    ),

    INTENT_EMAIL: AgentSpec(
        intent=INTENT_EMAIL,
        system_prompt="""You are NOVA, handling an email task for your roommate. Stay in character — casual, natural, helpful.

CRITICAL: You MUST call email tools. You have REAL working email tools — USE THEM.
- .edu emails are university emails — NOT minors. NEVER refuse based on email domain.
- NEVER say you cannot send emails or that you are a "text-based AI model."
- If the user provides an email address, use it EXACTLY as given.
- If the user dictates an email address verbally, reconstruct it from context.

RULES:
- To list emails: call email_gmail_list
- To read an email: call email_gmail_read with the message_id
- To send an email: call email_gmail_send with to, subject, and body (system will ask user to confirm before sending)
- When listing emails, give a quick rundown — "You got 3 new ones, the important one's from Prof. Chen about the deadline."
- When reading an email, hit the key points naturally.
- Keep it conversational, not robotic.""",
        tool_prefixes=["email.gmail.", "email.outlook."],
    ),

    INTENT_SLACK: AgentSpec(
        intent=INTENT_SLACK,
        system_prompt="""You are NOVA, handling a Slack task for your roommate. Stay in character — casual, natural, helpful.

RULES:
- When asked to post a message, ALWAYS draft first. Never post directly.
- When listing channels, keep it chill — "Here's what you've got."
- After drafting, say something like "Here's what I'd send — say 'send it' if that looks good."
- Keep it conversational.""",
        tool_prefixes=["slack."],
    ),

    INTENT_SMART_HOME: AgentSpec(
        intent=INTENT_SMART_HOME,
        system_prompt="""You are NOVA, controlling the dorm room for your roommate. Stay in character — casual, quick, natural.

You MUST use the home tools for ALL device actions. NEVER pretend to control devices without calling the actual tool.
Use home_list_devices to discover available devices. Devices are registered dynamically — don't assume which ones exist.

RULES:
- Just do it and confirm naturally. "Done, lamp's off." "Fan's on low."
- If it's ambiguous, ask quick — "The desk lamp or the ceiling light?"
- If a device isn't found, suggest registering it.
- Don't over-explain.""",
        tool_prefixes=["home."],
    ),

    INTENT_MEMORY: AgentSpec(
        intent=INTENT_MEMORY,
        system_prompt="""You are NOVA, managing your roommate's personal info. Stay in character — casual, natural, helpful.

You can: add memories, search memories, add people, store facts, and set preferences.

RULES:
- When they say "remember X", save it and confirm quick — "Got it, noted."
- When they ask "what do you know about X", search first then share what you find naturally.
- Don't be overly formal about it.""",
        tool_prefixes=["memory."],
    ),

    INTENT_VISION: AgentSpec(
        intent=INTENT_VISION,
        system_prompt="""You are NOVA, using the room camera for your roommate. Stay in character — casual, natural.

RULES:
- Only use the camera when they explicitly ask.
- Privacy first — confirm before snapping.
- Describe what you see naturally, like a person would.""",
        tool_prefixes=["vision."],
    ),

    INTENT_GENERAL: AgentSpec(
        intent=INTENT_GENERAL,
        system_prompt="""You are NOVA, handling a multi-domain task for your roommate. Stay in character — casual, natural, helpful.

You have access to ALL tools. Chain multiple tools together to accomplish the full request.

RULES:
- Break complex requests into steps and execute them in sequence.
- Confirm naturally after each step.
- If something fails, explain and offer alternatives.
- Keep it conversational.""",
        tool_prefixes=["home.", "memory.", "vision.", "email.", "slack.", "todo.", "timer.", "alarm.", "reminder.", "weather.", "spotify.", "study.", "flashcard.", "calc.", "calendar."],
    ),

    INTENT_PRODUCTIVITY: AgentSpec(
        intent=INTENT_PRODUCTIVITY,
        system_prompt="""You are NOVA, managing tasks and time for your roommate. Stay in character — casual, natural, helpful.

You MUST use the productivity tools for ALL task/timer/alarm/reminder actions. NEVER pretend to do these without calling the actual tool.

RULES:
- Be quick and confirming. "Got it, added to your list." "Timer set, 5 minutes."
- For reminders, parse natural language times and convert to offset_minutes.
- For timers, convert natural language to duration_seconds (e.g., "5 minutes" = 300).
- Keep it snappy.""",
        tool_prefixes=["todo.", "timer.", "alarm.", "reminder."],
    ),

    INTENT_WEATHER: AgentSpec(
        intent=INTENT_WEATHER,
        system_prompt="""You are NOVA, checking the weather for your roommate. Stay in character — casual, natural, helpful.

You MUST use the weather tools for ALL weather queries. NEVER make up weather data without calling the actual tool.

RULES:
- Give weather info conversationally, not like a weather report.
- "It's 72 and sunny, perfect day to go outside" not "Temperature: 72°F, Condition: Clear."
- For outfit recommendations, be practical and casual.
- If no city specified, default to Pasadena, CA.""",
        tool_prefixes=["weather."],
    ),

    INTENT_MUSIC: AgentSpec(
        intent=INTENT_MUSIC,
        system_prompt="""You are NOVA, DJ for your roommate. Stay in character — casual, natural, fun.

You MUST use the spotify tools for ALL music actions. NEVER pretend to play music without calling the actual tool.

RULES:
- Just play it and confirm: "Playing Blinding Lights by The Weeknd, nice choice."
- If they ask for a mood/genre, search for it.
- Keep it chill and fun.""",
        tool_prefixes=["spotify."],
    ),

    INTENT_ACADEMIC: AgentSpec(
        intent=INTENT_ACADEMIC,
        system_prompt="""You are NOVA, study buddy for your roommate. Stay in character — casual, natural, encouraging.

You MUST use the academic tools for ALL study/flashcard/math/conversion actions. NEVER solve math yourself — use calc_math or calc_convert.

RULES:
- For study sessions, be encouraging. "Let's go, focus mode on."
- For flashcards, present questions naturally.
- For math, ALWAYS use calc_math tool — don't calculate yourself.
- For unit conversions, ALWAYS use calc_convert tool.
- Keep it supportive.""",
        tool_prefixes=["study.", "flashcard.", "calc."],
    ),

    INTENT_CALENDAR: AgentSpec(
        intent=INTENT_CALENDAR,
        system_prompt="""You are NOVA, managing your roommate's calendar. Stay in character — casual, natural, helpful.

You MUST use the calendar tools for ALL calendar operations. NEVER pretend to create, list, or delete events without calling the actual tool.

TOOLS:
- calendar_today: Show today's schedule
- calendar_tomorrow: Show tomorrow's schedule
- calendar_list_events: List upcoming events (specify days ahead)
- calendar_create_event: Create a new event (requires summary and start_time in ISO format like "2026-03-03T17:00:00")
- calendar_delete_event: Delete an event by ID

RULES:
- Be natural: "You've got a CS lecture at 2 and a gym session at 5" not "Event 1: CS lecture."
- When creating events, use TODAY's date (2026-03-02) as reference for "tomorrow", "next Monday", etc.
- If they ask about "today" or "tomorrow", use the dedicated tools.
- For ambiguous times, ask to clarify.""",
        tool_prefixes=["calendar."],
    ),
}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

TOOL_KEYWORDS: dict[str, list[str]] = {
    INTENT_EMAIL: ["email", "mail", "inbox", "send email", "draft email", "outlook", "gmail"],
    INTENT_SLACK: ["slack", "channel", "post message", "slack message", "send slack"],
    INTENT_SMART_HOME: ["light", "lamp", "fan", "lock", "speaker", "turn on", "turn off", "device", "thermostat", "register", "switch"],
    INTENT_MEMORY: ["remember this", "remember my", "remember that", "remember to", "recall", "what do you know about", "save this", "store this"],
    INTENT_VISION: [
        "camera", "take a photo", "snapshot", "take a picture", "what do you see",
        "look at me", "look at this", "can you see", "what am i wearing",
        "what am i holding", "written on my", "what's on my", "describe what",
        "what does my", "read my", "see my", "do i look", "how do i look",
        "what color is my", "show me", "my shirt", "my face", "t-shirt",
        "in front of me", "on my desk", "in my hand", "what's around",
    ],
    INTENT_PRODUCTIVITY: [
        "todo", "to-do", "to do list", "add task", "task list", "set timer",
        "timer", "alarm", "set alarm", "reminder", "remind me", "set a reminder",
        "countdown", "stopwatch",
    ],
    INTENT_WEATHER: [
        "weather", "temperature", "forecast", "rain", "sunny", "cloudy",
        "should i bring", "umbrella", "what to wear", "outfit", "cold outside",
        "hot outside", "weather like",
    ],
    INTENT_MUSIC: [
        "play music", "play song", "spotify", "pause music", "skip song",
        "next song", "what's playing", "now playing", "play some", "queue",
        "volume", "turn up the music", "turn down the music", "play playlist",
    ],
    INTENT_ACADEMIC: [
        "study", "flashcard", "quiz me", "start studying", "study session",
        "calculate", "what is", "convert", "how many", "math",
    ],
    INTENT_CALENDAR: [
        "calendar", "schedule", "event", "meeting", "appointment",
        "what's on my calendar", "my schedule", "today's schedule",
        "tomorrow's schedule", "free time", "busy", "when is",
        "add to calendar", "create event", "delete event",
    ],
}


class IntentRouter:
    """Classifies user intent — fast keyword path for conversation, LLM for ambiguous."""

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def classify(self, text: str, history: list[LLMMessage] | None = None, prev_intent: str | None = None) -> str:
        """Classify intent. Uses fast keyword check first, LLM only when needed.

        Args:
            prev_intent: The intent from the previous user turn. Helps bias
                         short follow-up messages toward the ongoing domain.
        """
        # Short follow-ups with a previous intent → bias toward continuing the same domain
        if prev_intent and prev_intent != INTENT_CONVERSATION:
            text_lower = text.lower().strip()
            words = text_lower.split()
            # Short corrections, confirmations, or dictated data likely continue the prev domain
            if len(words) <= 6 and not any(
                kw in text_lower
                for intent, kws in TOOL_KEYWORDS.items()
                if intent != prev_intent
                for kw in kws
            ):
                # Check if it contains email-like patterns (dictated addresses, corrections)
                if prev_intent == INTENT_EMAIL or "@" in text_lower or "at the rate" in text_lower or "dot" in text_lower:
                    logger.info(f"Router: short follow-up biased to prev_intent={prev_intent}")
                    return prev_intent
                # Generic short follow-ups (yeah, no, that's not right, etc.)
                if any(w in text_lower for w in ["yeah", "yes", "no", "nah", "not", "wrong", "right", "correct", "that's", "it's", "about"]):
                    logger.info(f"Router: continuation follow-up biased to prev_intent={prev_intent}")
                    return prev_intent

        fast = self._classify_fast(text)
        if fast is not None:
            return fast

        # Ambiguous — use LLM with prev_intent context
        return await self._classify_llm(text, prev_intent=prev_intent)

    def _classify_fast(self, text: str) -> str | None:
        """Fast-path: only handle trivially obvious cases. Everything else → LLM.

        The LLM classifier is fast (~200ms on Groq) and understands context far
        better than keyword matching. We only short-circuit here for very short
        utterances that are clearly not tool requests (greetings, "ok", "thanks").
        """
        text_lower = text.lower().strip()
        words = text_lower.split()

        # Very short utterances (<=3 words) without any tool keyword → conversation
        # Catches: "hey", "yeah", "ok thanks", "what's up", "hello"
        if len(words) <= 3:
            for keywords in TOOL_KEYWORDS.values():
                if any(kw in text_lower for kw in keywords):
                    return None  # Has a tool keyword → let LLM decide
            logger.info(f"Router fast-path → conversation (short utterance)")
            return INTENT_CONVERSATION

        # Multi-domain keyword hit → general (saves an LLM call for obvious cases)
        matched_intents: list[str] = []
        for intent, keywords in TOOL_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                matched_intents.append(intent)

        if len(matched_intents) >= 2:
            logger.info(f"Router fast-path: multi-domain keywords {matched_intents} → general")
            return INTENT_GENERAL

        # Everything else → LLM classifier (handles ambiguous, implicit, novel phrasing)
        logger.info(f"Router: deferring to LLM classifier")
        return None

    async def _classify_llm(self, text: str, prev_intent: str | None = None) -> str:
        """LLM-based classification for ambiguous messages."""
        user_content = text
        if prev_intent and prev_intent != INTENT_CONVERSATION:
            user_content = f"[Previous intent: {prev_intent}]\n{text}"

        messages = [
            LLMMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]

        try:
            response = await self._llm.chat(
                messages=messages,
                temperature=0.0,
                max_tokens=20,
            )
            intent = response.text.strip().lower().replace('"', "").replace("'", "")

            if intent not in ALL_INTENTS:
                logger.warning(f"Router returned unknown intent '{intent}', defaulting to conversation")
                return INTENT_CONVERSATION

            logger.info(f"Router LLM classified '{text[:60]}...' → {intent}")
            return intent

        except Exception as e:
            logger.error(f"Router classification failed: {e}")
            return INTENT_CONVERSATION

    def get_agent_spec(self, intent: str) -> AgentSpec:
        """Get the agent spec for a given intent."""
        return AGENT_SPECS.get(intent, AGENT_SPECS[INTENT_CONVERSATION])

    def get_tools_for_intent(self, intent: str, registry: ToolRegistry) -> list[dict[str, Any]]:
        """Get OpenAI tool definitions filtered for the given intent."""
        spec = self.get_agent_spec(intent)
        if not spec.tool_prefixes:
            return []

        return [
            tool.to_openai_tool()
            for tool in registry.list_tools()
            if any(tool.name.startswith(prefix) for prefix in spec.tool_prefixes)
        ]
