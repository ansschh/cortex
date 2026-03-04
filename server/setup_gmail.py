"""Gmail OAuth setup — run once to authorize and store refresh token."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
]
TOKEN_PATH = "data/gmail_token.json"


def main():
    print("=" * 50)
    print("  NOVA — Gmail OAuth Setup")
    print("=" * 50)
    print()

    client_id = os.getenv("GMAIL_CLIENT_ID", "")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        print("ERROR: GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET must be set in .env")
        print()
        print("Steps to get these:")
        print("  1. Go to https://console.cloud.google.com/")
        print("  2. Create a project (or select existing)")
        print("  3. Enable the Gmail API")
        print("  4. Go to Credentials → Create Credentials → OAuth client ID")
        print("  5. Application type: Desktop app")
        print("  6. Copy client ID and secret into .env")
        sys.exit(1)

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost:8090"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

    print("Opening browser for Google OAuth authorization...")
    print("If the browser doesn't open, check the terminal for a URL.")
    print()

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=8090)

    os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())

    print()
    print("=" * 50)
    print("  Gmail authorization COMPLETE!")
    print(f"  Token saved to: {os.path.abspath(TOKEN_PATH)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
