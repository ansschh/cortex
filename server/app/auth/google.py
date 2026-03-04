"""Shared Google OAuth 2.0 — one login for Gmail, Calendar, and other Google APIs.

Web-based flow (not the "installed app" popup):
1. User visits /auth/google → redirected to Google consent
2. Google redirects back to /auth/google/callback with auth code
3. We exchange code for access + refresh tokens → stored in data/google_token.json
4. All Google tools (Gmail, Calendar) share the same credentials
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

logger = logging.getLogger(__name__)

TOKEN_PATH = Path("data/google_token.json")

# All scopes NOVA needs — requesting them all at once so user only logs in once
SCOPES = [
    # Gmail
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    # Calendar
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]


def _build_client_config(client_id: str, client_secret: str, redirect_uri: str) -> dict:
    """Build the client config dict for Google OAuth."""
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": [redirect_uri],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


def get_auth_url(client_id: str, client_secret: str, redirect_uri: str) -> str:
    """Generate the Google OAuth consent URL."""
    config = _build_client_config(client_id, client_secret, redirect_uri)
    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return auth_url


def exchange_code(code: str, client_id: str, client_secret: str, redirect_uri: str) -> Credentials:
    """Exchange the authorization code for credentials and save them."""
    config = _build_client_config(client_id, client_secret, redirect_uri)
    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)
    flow.fetch_token(code=code)
    creds = flow.credentials

    # Save token
    _save_credentials(creds)
    logger.info("Google OAuth tokens saved successfully")
    return creds


def get_credentials() -> Optional[Credentials]:
    """Load saved credentials, refreshing if expired. Returns None if not authenticated."""
    if not TOKEN_PATH.exists():
        return None

    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    except Exception as e:
        logger.error(f"Failed to load Google credentials: {e}")
        return None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                _save_credentials(creds)
                logger.info("Google token refreshed")
            except Exception as e:
                logger.error(f"Failed to refresh Google token: {e}")
                return None
        else:
            return None

    return creds


def is_authenticated() -> bool:
    """Check if we have valid Google credentials."""
    return get_credentials() is not None


def _save_credentials(creds: Credentials) -> None:
    """Save credentials to disk."""
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
