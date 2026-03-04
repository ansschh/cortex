"""Tool base class and registry — every tool inherits from BaseTool."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from shared.schemas.events import SensitivityLevel
from shared.schemas.tool_calls import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# Patterns that indicate a missing or placeholder API key
_EMPTY_KEY_PATTERNS = ("", "...", "xoxb-...", "sk-...", "your-", "TODO", "CHANGEME")

# Map tool name prefixes → human-friendly service names + env var hints
_SERVICE_HINTS: dict[str, tuple[str, str]] = {
    "weather.":   ("OpenWeatherMap", "OPENWEATHERMAP_API_KEY"),
    "spotify.":   ("Spotify", "SPOTIFY_ACCESS_TOKEN"),
    "gmail.":     ("Gmail", "GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET"),
    "outlook.":   ("Microsoft Outlook", "MS_CLIENT_ID / MS_CLIENT_SECRET"),
    "slack.":     ("Slack", "SLACK_BOT_TOKEN"),
    "calendar.":  ("Google Calendar", "GOOGLE_CALENDAR_CREDENTIALS"),
    "discord.":   ("Discord", "DISCORD_BOT_TOKEN"),
    "perplexity.": ("Perplexity", "PERPLEXITY_API_KEY"),
    "tavily.":    ("Tavily", "TAVILY_API_KEY"),
    "genius.":    ("Genius", "GENIUS_ACCESS_TOKEN"),
    "tmdb.":      ("TMDB", "TMDB_API_KEY"),
    "google_search.": ("Google Custom Search", "GOOGLE_CUSTOM_SEARCH_API_KEY"),
    "maps.":      ("Google Maps", "GOOGLE_MAPS_API_KEY"),
    "openai.":    ("OpenAI / OpenRouter", "OPENAI_API_KEY"),
    "news.":      ("News API", "NEWSAPI_KEY"),
    "wolfram.":   ("Wolfram Alpha", "WOLFRAM_ALPHA_APP_ID"),
    "notion.":    ("Notion", "NOTION_API_KEY"),
    "todoist.":   ("Todoist", "TODOIST_API_KEY"),
    "youtube.":   ("YouTube / RapidAPI", "RAPIDAPI_KEY"),
    "shazam.":    ("Shazam / RapidAPI", "RAPIDAPI_KEY"),
    "uber.":      ("Uber", "UBER_API_TOKEN"),
    "fitbit.":    ("Fitbit", "FITBIT_ACCESS_TOKEN"),
    "reddit.":    ("Reddit", "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET"),
    "twitter.":   ("Twitter/X", "TWITTER_BEARER_TOKEN"),
    "telegram.":  ("Telegram", "TELEGRAM_BOT_TOKEN"),
    "twilio.":    ("Twilio", "TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN"),
    "whatsapp.":  ("WhatsApp / Twilio", "TWILIO_ACCOUNT_SID"),
    "github.":    ("GitHub", "GITHUB_TOKEN"),
    "gitlab.":    ("GitLab", "GITLAB_TOKEN"),
    "nasa.":      ("NASA", "NASA_API_KEY"),
    "yelp.":      ("Yelp", "YELP_API_KEY"),
    "nutrition.": ("Nutritionix", "NUTRITIONIX_API_KEY"),
    "spoonacular.": ("Spoonacular", "SPOONACULAR_API_KEY"),
}


def _get_service_hint(tool_name: str) -> tuple[str, str]:
    """Look up (service_name, env_var) for a tool name."""
    for prefix, hint in _SERVICE_HINTS.items():
        if tool_name.startswith(prefix):
            return hint
    # Fallback: derive from tool name
    service = tool_name.split(".")[0].replace("_", " ").title()
    env_var = tool_name.split(".")[0].upper() + "_API_KEY"
    return service, env_var


def _friendly_error(tool_name: str, error: Exception) -> str:
    """Turn a raw exception into a friendly, informative error message for the LLM."""
    err_str = str(error).lower()
    service_name, env_var = _get_service_hint(tool_name)

    # Missing / empty API key
    if any(phrase in err_str for phrase in ("api key", "not configured", "not set", "token", "unauthorized", "forbidden")):
        return (
            f"[MISSING_API_KEY] The {service_name} integration is not set up yet. "
            f"The user needs to add their API key to the environment variable '{env_var}'. "
            f"They can do this at http://localhost:8000/integrations or by editing the .env file. "
            f"Tell the user in a friendly way that this service isn't connected yet and how to fix it."
        )

    # HTTP 401/403 — bad or expired key
    if "401" in err_str or "403" in err_str or "invalid_token" in err_str or "expired" in err_str:
        return (
            f"[INVALID_API_KEY] The {service_name} API key appears to be invalid or expired. "
            f"The env var is '{env_var}'. The user should get a fresh key from the {service_name} dashboard. "
            f"Tell the user their API key for {service_name} seems broken and they should regenerate it."
        )

    # HTTP 429 — rate limit
    if "429" in err_str or "rate limit" in err_str or "too many" in err_str:
        return (
            f"[RATE_LIMITED] {service_name} is rate-limiting us — too many requests. "
            f"Tell the user to wait a bit and try again, or that we've hit the API's free tier limit."
        )

    # Network / timeout errors
    if any(phrase in err_str for phrase in ("timeout", "timed out", "connect", "connectionerror", "dns", "unreachable")):
        return (
            f"[NETWORK_ERROR] Couldn't reach the {service_name} API — looks like a network or timeout issue. "
            f"Tell the user there might be a connectivity problem and to check their internet."
        )

    # Generic — still make it informative
    return (
        f"[TOOL_ERROR] The {service_name} tool ({tool_name}) hit an error: {error}. "
        f"Tell the user what went wrong in a helpful way. If it looks like a config issue, "
        f"mention the env var '{env_var}' and the setup page at http://localhost:8000/integrations."
    )


class BaseTool(ABC):
    """Interface every tool must implement."""

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = {}
    requires_confirmation: bool = False
    sensitivity: SensitivityLevel = SensitivityLevel.LOW

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
            requires_confirmation=self.requires_confirmation,
            sensitivity=self.sensitivity,
        )

    @property
    def llm_name(self) -> str:
        """Tool name safe for LLMs (dots replaced with underscores).

        Llama/Groq models choke on dots in function names, producing
        malformed tool calls like <function=email.gmail.list{...}>.
        """
        return self.name.replace(".", "_")

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format for the LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.llm_name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        ...

    async def safe_execute(self, **kwargs: Any) -> ToolResult:
        """Wraps execute() with friendly error handling for API key / network issues."""
        try:
            result = await self.execute(**kwargs)
            if not result.success:
                result = self._enrich_error(result)
            return result
        except Exception as e:
            logger.error(f"Tool {self.name} error: {e}")
            friendly = _friendly_error(self.name, e)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=friendly,
                result={"error": friendly},
            )

    def _enrich_error(self, result: ToolResult) -> ToolResult:
        """Enrich a failed ToolResult with friendly, tagged error messages."""
        # Collect the raw error text from wherever it lives
        raw_error = ""
        if result.error:
            raw_error = result.error
        elif isinstance(result.result, dict) and "error" in result.result:
            raw_error = str(result.result["error"])

        if not raw_error:
            return result

        err_lower = raw_error.lower()
        service_name, env_var = _get_service_hint(self.name)

        # Already tagged — don't double-enrich
        if raw_error.startswith("["):
            return result

        # Detect error category and build friendly message
        _KEY_PHRASES = ("api key", "not configured", "not set", "not connected", "no token", "missing key")
        _AUTH_PHRASES = ("401", "403", "unauthorized", "forbidden", "invalid_token", "expired", "invalid key")
        _RATE_PHRASES = ("429", "rate limit", "too many requests", "quota")
        _NET_PHRASES = ("timeout", "timed out", "connection", "unreachable", "dns", "network")

        if any(p in err_lower for p in _KEY_PHRASES):
            friendly = (
                f"[MISSING_API_KEY] The {service_name} integration isn't set up yet. "
                f"The user needs to add their API key to '{env_var}'. "
                f"They can set it up at http://localhost:8000/integrations or in the .env file."
            )
        elif any(p in err_lower for p in _AUTH_PHRASES):
            friendly = (
                f"[INVALID_API_KEY] The {service_name} API key seems invalid or expired. "
                f"The env var is '{env_var}'. User should get a fresh key from {service_name}'s dashboard."
            )
        elif any(p in err_lower for p in _RATE_PHRASES):
            friendly = (
                f"[RATE_LIMITED] {service_name} is rate-limiting us. "
                f"Tell the user to wait a moment and try again."
            )
        elif any(p in err_lower for p in _NET_PHRASES):
            friendly = (
                f"[NETWORK_ERROR] Can't reach the {service_name} API right now. "
                f"Might be a connectivity issue."
            )
        else:
            # Keep original error but wrap it for context
            friendly = (
                f"[TOOL_ERROR] {service_name} tool error: {raw_error}. "
                f"If this looks like a config issue, the env var is '{env_var}'."
            )

        result.error = friendly
        if isinstance(result.result, dict):
            result.result["error"] = friendly
        return result


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._llm_name_map: dict[str, BaseTool] = {}  # underscored → tool

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        self._llm_name_map[tool.llm_name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Lookup by canonical name (dotted) or LLM name (underscored)."""
        tool = self._tools.get(name)
        if tool:
            return tool
        # Try LLM name (underscored version used by function calling)
        return self._llm_name_map.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai_tool() for t in self._tools.values()]

    def get_definitions(self) -> list[ToolDefinition]:
        return [t.get_definition() for t in self._tools.values()]
