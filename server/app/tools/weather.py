"""Weather tools — current conditions, forecasts, alerts, outfit suggestions."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from server.app.tools.base import BaseTool
from shared.schemas.tool_calls import ToolResult

logger = logging.getLogger(__name__)

# OpenWeatherMap free tier: 1000 calls/day
_OWM_BASE = "https://api.openweathermap.org/data/2.5"


class WeatherCurrentTool(BaseTool):
    name = "weather.current"
    description = "Get current weather conditions for a location."
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name (e.g., 'New York' or 'London,UK')"},
        },
        "required": ["city"],
    }

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def execute(self, **kwargs: Any) -> ToolResult:
        city = kwargs["city"]
        if not self._api_key:
            return ToolResult(tool_name=self.name, success=False, result={"error": "OpenWeatherMap API key not configured"})

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_OWM_BASE}/weather", params={
                "q": city, "appid": self._api_key, "units": "imperial",
            })

        if resp.status_code != 200:
            return ToolResult(tool_name=self.name, success=False, result={"error": f"Weather API error: {resp.text}"})

        data = resp.json()
        weather = {
            "city": data.get("name", city),
            "temp_f": data["main"]["temp"],
            "feels_like_f": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_mph": data["wind"]["speed"],
        }
        return ToolResult(tool_name=self.name, success=True, result=weather)


class WeatherForecastTool(BaseTool):
    name = "weather.forecast"
    description = "Get a 5-day weather forecast for a location."
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "description": "Number of days (1-5, default 3)"},
        },
        "required": ["city"],
    }

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def execute(self, **kwargs: Any) -> ToolResult:
        city = kwargs["city"]
        days = min(int(kwargs.get("days", 3)), 5)

        if not self._api_key:
            return ToolResult(tool_name=self.name, success=False, result={"error": "OpenWeatherMap API key not configured"})

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_OWM_BASE}/forecast", params={
                "q": city, "appid": self._api_key, "units": "imperial", "cnt": days * 8,
            })

        if resp.status_code != 200:
            return ToolResult(tool_name=self.name, success=False, result={"error": f"Forecast API error: {resp.text}"})

        data = resp.json()
        forecasts = []
        seen_dates = set()
        for item in data.get("list", []):
            date = item["dt_txt"].split(" ")[0]
            if date not in seen_dates and "12:00:00" in item["dt_txt"]:
                seen_dates.add(date)
                forecasts.append({
                    "date": date,
                    "temp_f": item["main"]["temp"],
                    "description": item["weather"][0]["description"],
                    "humidity": item["main"]["humidity"],
                    "wind_mph": item["wind"]["speed"],
                })
                if len(forecasts) >= days:
                    break

        return ToolResult(tool_name=self.name, success=True, result={"city": city, "forecasts": forecasts})


class WeatherHourlyTool(BaseTool):
    name = "weather.hourly"
    description = "Get hourly weather forecast for the next 12 hours."
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    }

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def execute(self, **kwargs: Any) -> ToolResult:
        city = kwargs["city"]
        if not self._api_key:
            return ToolResult(tool_name=self.name, success=False, result={"error": "API key not configured"})

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_OWM_BASE}/forecast", params={
                "q": city, "appid": self._api_key, "units": "imperial", "cnt": 4,
            })

        if resp.status_code != 200:
            return ToolResult(tool_name=self.name, success=False, result={"error": resp.text})

        data = resp.json()
        hours = []
        for item in data.get("list", [])[:4]:
            hours.append({
                "time": item["dt_txt"],
                "temp_f": item["main"]["temp"],
                "description": item["weather"][0]["description"],
                "rain_chance": item.get("pop", 0) * 100,
            })

        return ToolResult(tool_name=self.name, success=True, result={"city": city, "hourly": hours})


class WeatherAlertsTool(BaseTool):
    name = "weather.alerts"
    description = "Check for severe weather alerts (US only, uses NWS API)."
    parameters_schema = {
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude"},
            "lon": {"type": "number", "description": "Longitude"},
            "state": {"type": "string", "description": "US state code (e.g., 'NY', 'CA') — alternative to lat/lon"},
        },
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        state = kwargs.get("state")
        lat = kwargs.get("lat")
        lon = kwargs.get("lon")

        async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "NOVA-Assistant"}) as client:
            if state:
                resp = await client.get(f"https://api.weather.gov/alerts/active?area={state}")
            elif lat and lon:
                resp = await client.get(f"https://api.weather.gov/alerts/active?point={lat},{lon}")
            else:
                return ToolResult(tool_name=self.name, success=False, result={"error": "Provide state code or lat/lon"})

        if resp.status_code != 200:
            return ToolResult(tool_name=self.name, success=False, result={"error": "NWS API error"})

        data = resp.json()
        alerts = []
        for feature in data.get("features", [])[:5]:
            props = feature.get("properties", {})
            alerts.append({
                "event": props.get("event", ""),
                "severity": props.get("severity", ""),
                "headline": props.get("headline", ""),
                "description": (props.get("description", ""))[:200],
            })

        return ToolResult(tool_name=self.name, success=True, result={"alerts": alerts, "count": len(alerts)})


class WeatherOutfitTool(BaseTool):
    name = "weather.outfit"
    description = "Get clothing recommendation based on current weather."
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    }

    def __init__(self, api_key: str, llm=None):
        self._api_key = api_key
        self._llm = llm

    async def execute(self, **kwargs: Any) -> ToolResult:
        city = kwargs["city"]
        if not self._api_key:
            return ToolResult(tool_name=self.name, success=False, result={"error": "API key not configured"})

        # First get weather
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_OWM_BASE}/weather", params={
                "q": city, "appid": self._api_key, "units": "imperial",
            })

        if resp.status_code != 200:
            return ToolResult(tool_name=self.name, success=False, result={"error": resp.text})

        data = resp.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        wind = data["wind"]["speed"]

        # Simple rule-based outfit recommendation (no LLM needed)
        layers = []
        if temp < 32:
            layers = ["heavy winter coat", "warm hat", "gloves", "scarf", "boots"]
        elif temp < 50:
            layers = ["jacket or heavy sweater", "long pants", "closed-toe shoes"]
        elif temp < 65:
            layers = ["light jacket or hoodie", "long pants"]
        elif temp < 80:
            layers = ["t-shirt", "jeans or shorts"]
        else:
            layers = ["light clothing", "shorts", "sandals"]

        if "rain" in desc.lower():
            layers.append("umbrella")
            layers.append("waterproof shoes")
        if wind > 15:
            layers.append("windbreaker")

        return ToolResult(
            tool_name=self.name, success=True,
            result={
                "city": city, "temp_f": temp, "conditions": desc,
                "recommendation": layers,
            },
        )
