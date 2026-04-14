"""
Full MCP smoke test — calls all four tools via stdio client.
Run with: python smoke_test.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

URL = (
    sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=gdx7gN1UyX0"
)
JOB_ID = sys.argv[2] if len(sys.argv) > 2 else f"sig_{os.urandom(4).hex()}"

# Ensure ffmpeg and venv-installed tools (yt-dlp, demucs) are on PATH
# for subprocesses spawned by the server.
VENV_SCRIPTS = str(Path(__file__).parent / ".venv" / "Scripts")
FFMPEG_BIN = (
    r"C:\Users\ROOP\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin"
)
_env = os.environ.copy()
_env["PATH"] = (
    VENV_SCRIPTS + os.pathsep + FFMPEG_BIN + os.pathsep + _env.get("PATH", "")
)


def print_result(tool: str, result):
    print(f"\n{'=' * 60}")
    print(f"  {tool}")
    print(f"{'=' * 60}")
    for content in result.content:
        try:
            parsed = json.loads(content.text)
            print(json.dumps(parsed, indent=2))
        except Exception:
            print(content.text)


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=_env,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to server.\n")

            # 1. check_health
            print(">>> check_health")
            result = await session.call_tool("check_health", {})
            print_result("check_health", result)

            # 2. list_jobs (empty at start)
            print("\n>>> list_jobs (before split)")
            result = await session.call_tool("list_jobs", {})
            print_result("list_jobs", result)

            # 3. split_audio (Fast Resume)
            print(f"\n>>> split_audio  url={URL} job_id={JOB_ID}")
            print("    (Starting full production pipeline...)")
            result = await asyncio.wait_for(
                session.call_tool(
                    "split_audio", {"url": URL, "job_id": JOB_ID, "model": "htdemucs"}
                ),
                timeout=3600,
            )
            print_result("split_audio", result)

            # Extract job_id from response
            job_id = None
            for content in result.content:
                try:
                    payload = json.loads(content.text)
                    job_id = payload.get("header", {}).get("job_id")
                except Exception:
                    pass

            # 4. get_job_status
            if job_id:
                print(f"\n>>> get_job_status  job_id={job_id}")
                result = await session.call_tool("get_job_status", {"job_id": job_id})
                print_result("get_job_status", result)

            # 5. list_jobs (should show completed job)
            print("\n>>> list_jobs (after split)")
            result = await session.call_tool("list_jobs", {})
            print_result("list_jobs", result)


asyncio.run(main())
