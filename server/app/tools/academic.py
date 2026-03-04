"""Academic tools — study sessions, flashcards, math, unit conversion."""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta
from typing import Any

import aiosqlite

from server.app.tools.base import BaseTool
from shared.schemas.tool_calls import ToolResult


# ------------------------------------------------------------------
# Study Sessions (Pomodoro-style)
# ------------------------------------------------------------------

class StudyStartTool(BaseTool):
    name = "study.start_session"
    description = "Start a study/focus session. Optionally specify a subject."
    parameters_schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "What you're studying (e.g., 'Calculus', 'CS 101')"},
        },
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        subject = kwargs.get("subject", "General")
        now = time.time()

        cursor = await self._db.execute(
            "INSERT INTO study_sessions (subject, start_time) VALUES (?, ?)",
            (subject, now),
        )
        await self._db.commit()
        return ToolResult(
            tool_name=self.name, success=True,
            result={
                "session_id": cursor.lastrowid,
                "subject": subject,
                "started_at": datetime.fromtimestamp(now).strftime("%H:%M"),
            },
        )


class StudyEndTool(BaseTool):
    name = "study.end_session"
    description = "End the current study session and log the duration."
    parameters_schema = {
        "type": "object",
        "properties": {
            "notes": {"type": "string", "description": "Optional notes about the session"},
        },
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        notes = kwargs.get("notes", "")
        now = time.time()

        # Find the most recent active session
        row = await self._db.execute_fetchall(
            "SELECT id, subject, start_time FROM study_sessions WHERE end_time IS NULL ORDER BY start_time DESC LIMIT 1"
        )
        if not row:
            return ToolResult(tool_name=self.name, success=False, result={"error": "No active study session"})

        session_id, subject, start_time = row[0]
        duration = now - start_time

        await self._db.execute(
            "UPDATE study_sessions SET end_time = ?, duration_seconds = ?, notes = ? WHERE id = ?",
            (now, duration, notes, session_id),
        )
        await self._db.commit()

        minutes = int(duration // 60)
        return ToolResult(
            tool_name=self.name, success=True,
            result={
                "session_id": session_id,
                "subject": subject,
                "duration_minutes": minutes,
                "notes": notes,
            },
        )


class StudyStatsTool(BaseTool):
    name = "study.stats"
    description = "Show study time statistics — today, this week, by subject."
    parameters_schema = {
        "type": "object",
        "properties": {
            "period": {"type": "string", "enum": ["today", "week", "month", "all"], "description": "Time period"},
        },
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        period = kwargs.get("period", "today")
        now = time.time()

        if period == "today":
            start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        elif period == "week":
            start = now - 7 * 86400
        elif period == "month":
            start = now - 30 * 86400
        else:
            start = 0

        rows = await self._db.execute_fetchall(
            "SELECT subject, SUM(duration_seconds) FROM study_sessions "
            "WHERE start_time >= ? AND duration_seconds IS NOT NULL GROUP BY subject",
            (start,),
        )

        by_subject = {}
        total = 0.0
        for subject, dur in rows:
            minutes = round(dur / 60, 1)
            by_subject[subject] = minutes
            total += dur

        return ToolResult(
            tool_name=self.name, success=True,
            result={
                "period": period,
                "total_minutes": round(total / 60, 1),
                "total_hours": round(total / 3600, 1),
                "by_subject": by_subject,
            },
        )


# ------------------------------------------------------------------
# Flashcards
# ------------------------------------------------------------------

class FlashcardCreateTool(BaseTool):
    name = "flashcard.create"
    description = "Create a new flashcard in a deck."
    parameters_schema = {
        "type": "object",
        "properties": {
            "deck": {"type": "string", "description": "Deck name (e.g., 'Biology 101')"},
            "front": {"type": "string", "description": "Question or prompt"},
            "back": {"type": "string", "description": "Answer"},
        },
        "required": ["deck", "front", "back"],
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        deck = kwargs["deck"]
        front = kwargs["front"]
        back = kwargs["back"]

        cursor = await self._db.execute(
            "INSERT INTO flashcards (deck, front, back) VALUES (?, ?, ?)",
            (deck, front, back),
        )
        await self._db.commit()
        return ToolResult(
            tool_name=self.name, success=True,
            result={"card_id": cursor.lastrowid, "deck": deck, "front": front},
        )


class FlashcardQuizTool(BaseTool):
    name = "flashcard.quiz"
    description = "Get a random flashcard to quiz yourself. Returns the question — you answer, then ask to check."
    parameters_schema = {
        "type": "object",
        "properties": {
            "deck": {"type": "string", "description": "Deck name to quiz from (or 'all')"},
        },
    }

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        deck = kwargs.get("deck", "all")

        if deck == "all":
            rows = await self._db.execute_fetchall(
                "SELECT id, deck, front, back FROM flashcards ORDER BY RANDOM() LIMIT 1"
            )
        else:
            rows = await self._db.execute_fetchall(
                "SELECT id, deck, front, back FROM flashcards WHERE deck = ? ORDER BY RANDOM() LIMIT 1",
                (deck,),
            )

        if not rows:
            return ToolResult(tool_name=self.name, success=False, result={"error": f"No flashcards in deck '{deck}'"})

        card_id, deck_name, front, back = rows[0]
        return ToolResult(
            tool_name=self.name, success=True,
            result={"card_id": card_id, "deck": deck_name, "question": front, "answer": back},
        )


class FlashcardListDecksTool(BaseTool):
    name = "flashcard.list_decks"
    description = "List all flashcard decks with card counts."
    parameters_schema = {"type": "object", "properties": {}}

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        rows = await self._db.execute_fetchall(
            "SELECT deck, COUNT(*) as card_count FROM flashcards GROUP BY deck ORDER BY deck"
        )
        decks = [{"deck": r[0], "card_count": r[1]} for r in rows]
        return ToolResult(tool_name=self.name, success=True, result={"decks": decks, "total_decks": len(decks)})


# ------------------------------------------------------------------
# Calculator & Unit Conversion
# ------------------------------------------------------------------

class CalcMathTool(BaseTool):
    name = "calc.math"
    description = "Evaluate a mathematical expression. Supports basic math, trig, logarithms, etc."
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression (e.g., 'sqrt(144) + 3^2', '2*pi*5', 'log(100)')"},
        },
        "required": ["expression"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        import math

        expr = kwargs["expression"]

        # Safe math evaluation — only allow math functions
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max,
            "pi": math.pi, "e": math.e, "tau": math.tau,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "log2": math.log2,
            "exp": math.exp, "pow": math.pow, "ceil": math.ceil, "floor": math.floor,
            "radians": math.radians, "degrees": math.degrees,
            "factorial": math.factorial, "gcd": math.gcd,
        }

        try:
            # Replace ^ with ** for exponentiation
            safe_expr = expr.replace("^", "**")
            result = eval(safe_expr, {"__builtins__": {}}, allowed)
            return ToolResult(
                tool_name=self.name, success=True,
                result={"expression": expr, "result": result},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, result={"error": str(e), "expression": expr})


class CalcConvertTool(BaseTool):
    name = "calc.convert"
    description = "Convert between units (temperature, distance, weight, volume, etc.)."
    parameters_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The numeric value to convert"},
            "from_unit": {"type": "string", "description": "Source unit (e.g., 'km', 'lbs', 'celsius')"},
            "to_unit": {"type": "string", "description": "Target unit (e.g., 'miles', 'kg', 'fahrenheit')"},
        },
        "required": ["value", "from_unit", "to_unit"],
    }

    # Simple conversion table — no external deps needed
    _CONVERSIONS = {
        # Length
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("m", "ft"): lambda x: x * 3.28084,
        ("ft", "m"): lambda x: x * 0.3048,
        ("cm", "inches"): lambda x: x * 0.393701,
        ("inches", "cm"): lambda x: x * 2.54,
        ("m", "cm"): lambda x: x * 100,
        ("cm", "m"): lambda x: x / 100,
        # Weight
        ("kg", "lbs"): lambda x: x * 2.20462,
        ("lbs", "kg"): lambda x: x * 0.453592,
        ("g", "oz"): lambda x: x * 0.035274,
        ("oz", "g"): lambda x: x * 28.3495,
        # Temperature
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("celsius", "kelvin"): lambda x: x + 273.15,
        ("kelvin", "celsius"): lambda x: x - 273.15,
        # Volume
        ("liters", "gallons"): lambda x: x * 0.264172,
        ("gallons", "liters"): lambda x: x * 3.78541,
        ("ml", "fl_oz"): lambda x: x * 0.033814,
        ("fl_oz", "ml"): lambda x: x * 29.5735,
        ("cups", "ml"): lambda x: x * 236.588,
        ("ml", "cups"): lambda x: x / 236.588,
        # Speed
        ("mph", "kph"): lambda x: x * 1.60934,
        ("kph", "mph"): lambda x: x * 0.621371,
    }

    # Aliases so the LLM can use natural names
    _ALIASES = {
        "kilometers": "km", "kilometer": "km",
        "meters": "m", "meter": "m",
        "centimeters": "cm", "centimeter": "cm",
        "feet": "ft", "foot": "ft",
        "inch": "inches",
        "pounds": "lbs", "pound": "lbs", "lb": "lbs",
        "kilograms": "kg", "kilogram": "kg",
        "grams": "g", "gram": "g",
        "ounces": "oz", "ounce": "oz",
        "c": "celsius", "f": "fahrenheit", "k": "kelvin",
        "liter": "liters", "l": "liters",
        "gallon": "gallons", "gal": "gallons",
        "milliliters": "ml", "milliliter": "ml",
        "fluid_oz": "fl_oz", "fluid_ounces": "fl_oz",
        "cup": "cups",
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        value = float(kwargs["value"])
        from_u = kwargs["from_unit"].lower().strip()
        to_u = kwargs["to_unit"].lower().strip()
        from_u = self._ALIASES.get(from_u, from_u)
        to_u = self._ALIASES.get(to_u, to_u)

        key = (from_u, to_u)
        if key in self._CONVERSIONS:
            result = self._CONVERSIONS[key](value)
            return ToolResult(
                tool_name=self.name, success=True,
                result={"value": value, "from": from_u, "to": to_u, "result": round(result, 4)},
            )
        else:
            available = [f"{f} → {t}" for f, t in self._CONVERSIONS]
            return ToolResult(
                tool_name=self.name, success=False,
                result={"error": f"Unknown conversion: {from_u} → {to_u}", "available": available},
            )
