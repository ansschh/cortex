"""Pipecat-derived turn detection for NOVA.

Uses the Smart Turn v3 ONNX model from pipecat-ai/pipecat (BSD 2-Clause License).
The ML model is the primary gate for end-of-turn detection — if it says INCOMPLETE,
the turn never ends regardless of pause length (up to the 3-second hard timeout).
"""

from client.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from client.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

__all__ = ["BaseTurnAnalyzer", "EndOfTurnState", "LocalSmartTurnAnalyzerV3"]
