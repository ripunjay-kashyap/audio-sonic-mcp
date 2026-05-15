"""
Tests for server.py dispatcher logic.
All pipeline stages are mocked — no audio processing, no network calls.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

import server
from server import get_sonic_signature, get_job_status, JOB_STORE


@pytest.fixture(autouse=True)
def clear_job_store():
    """Isolate JOB_STORE state between tests."""
    JOB_STORE.clear()
    yield
    JOB_STORE.clear()


# ── Fast return ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_sonic_signature_returns_queued_immediately():
    """Valid URL → returns queued job_id without running the pipeline."""
    with patch("server._run_pipeline", new=AsyncMock()):
        result = await get_sonic_signature("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    data = json.loads(result)
    assert data["status"] == "queued"
    assert "job_id" in data
    assert "message" in data


@pytest.mark.asyncio
async def test_get_sonic_signature_uses_provided_job_id():
    with patch("server._run_pipeline", new=AsyncMock()):
        result = await get_sonic_signature(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            job_id="my_custom_id",
        )

    data = json.loads(result)
    assert data["job_id"] == "my_custom_id"


@pytest.mark.asyncio
async def test_get_sonic_signature_stores_queued_status():
    with patch("server._run_pipeline", new=AsyncMock()):
        await get_sonic_signature(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            job_id="test_q",
        )

    assert JOB_STORE["test_q"]["status"] == "queued"


# ── URL validation ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_url_returns_error_immediately():
    """Bad URL → returns error JSON without touching JOB_STORE."""
    result = await get_sonic_signature("not-a-url")
    data = json.loads(result)
    assert data["header"]["status"] == "error"


@pytest.mark.asyncio
async def test_unsupported_domain_returns_error_immediately():
    result = await get_sonic_signature("https://vimeo.com/123456")
    data = json.loads(result)
    assert data["header"]["status"] == "error"


@pytest.mark.asyncio
async def test_invalid_url_not_added_to_job_store():
    await get_sonic_signature("not-a-url", job_id="bad_job")
    assert "bad_job" not in JOB_STORE or JOB_STORE.get("bad_job", {}).get("status") == "error"


# ── get_job_status ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_job_status_returns_queued_before_pipeline_runs():
    with patch("server._run_pipeline", new=AsyncMock()):
        await get_sonic_signature(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            job_id="poll_test",
        )

    status_result = await get_job_status("poll_test")
    data = json.loads(status_result)
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_get_job_status_unknown_job():
    result = await get_job_status("nonexistent_job")
    data = json.loads(result)
    assert "error" in data
