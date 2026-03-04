"""Convenience script to start the NOVA client from the project root."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from client.client_main import AssistantClient


async def main():
    client = AssistantClient()
    try:
        await client.start()
    except KeyboardInterrupt:
        pass
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
