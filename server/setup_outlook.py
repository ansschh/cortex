"""Outlook / Microsoft Graph OAuth setup — run once to authorize via device code flow."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import msal

SCOPES = ["Mail.Read", "Mail.Send"]
TOKEN_CACHE_PATH = "data/outlook_token_cache.json"


def main():
    print("=" * 50)
    print("  NOVA — Outlook / Microsoft 365 OAuth Setup")
    print("=" * 50)
    print()

    tenant_id = os.getenv("MS_TENANT_ID", "")
    client_id = os.getenv("MS_CLIENT_ID", "")

    if not tenant_id or not client_id:
        print("ERROR: MS_TENANT_ID and MS_CLIENT_ID must be set in .env")
        print()
        print("Steps to get these:")
        print("  1. Go to https://portal.azure.com/")
        print("  2. Azure Active Directory → App registrations → New registration")
        print("  3. Name: NOVA Assistant")
        print("  4. Supported account types: Accounts in this organizational directory only")
        print("  5. Redirect URI: leave blank (we use device code flow)")
        print("  6. After creation, copy Application (client) ID → MS_CLIENT_ID")
        print("  7. Copy Directory (tenant) ID → MS_TENANT_ID")
        print("  8. Go to API permissions → Add → Microsoft Graph → Delegated:")
        print("     - Mail.Read")
        print("     - Mail.Send")
        print("  9. Under Authentication → Allow public client flows: YES")
        sys.exit(1)

    cache = msal.SerializableTokenCache()

    app = msal.PublicClientApplication(
        client_id,
        authority=f"https://login.microsoftonline.com/{tenant_id}",
        token_cache=cache,
    )

    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        print(f"ERROR: Device flow failed: {json.dumps(flow, indent=2)}")
        sys.exit(1)

    print(f"Go to: {flow['verification_uri']}")
    print(f"Enter code: {flow['user_code']}")
    print()
    print("Waiting for authorization...")

    result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        print(f"ERROR: {result.get('error_description', 'Unknown error')}")
        sys.exit(1)

    os.makedirs(os.path.dirname(TOKEN_CACHE_PATH), exist_ok=True)
    with open(TOKEN_CACHE_PATH, "w") as f:
        f.write(cache.serialize())

    print()
    print("=" * 50)
    print("  Outlook authorization COMPLETE!")
    print(f"  Token cache saved to: {os.path.abspath(TOKEN_CACHE_PATH)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
