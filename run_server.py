"""Convenience script to start the NOVA server from the project root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import uvicorn
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from server.app.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.assistant_name} server on {settings.server_host}:{settings.server_port}")
    print(f"  UI:     http://localhost:{settings.server_port}/ui/")
    print(f"  Health: http://localhost:{settings.server_port}/health")
    print(f"  Chat:   POST http://localhost:{settings.server_port}/api/chat")
    print()
    uvicorn.run(
        "server.app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
        reload_dirs=[str(ROOT / "server"), str(ROOT / "shared")],
        reload_excludes=["client/*", "ui/*", "data/*", "*.log"],
    )
