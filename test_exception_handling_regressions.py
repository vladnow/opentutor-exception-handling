"""Expected failures degrade gracefully; programming errors remain visible."""

import json
import logging
import sys
import types
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zipfile import BadZipFile

import httpx
import pytest
from sqlalchemy.exc import SQLAlchemyError

from libs.exceptions import ExternalServiceError, ValidationError
from libs.text_utils import parse_llm_json
from routers import preferences_llm
from services.block_decision import preference
from services.browser import automation
from services.ingestion.document_loader_formats import _extract_xlsx_fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["record", "onboarding", "preferences"])
async def test_preference_database_failures_are_logged(operation, caplog):
    user_id, course_id = uuid.uuid4(), uuid.uuid4()
    db = SimpleNamespace(
        add=lambda _: None,
        flush=AsyncMock(side_effect=SQLAlchemyError("database unavailable")),
        execute=AsyncMock(side_effect=SQLAlchemyError("database unavailable")),
    )
    with caplog.at_level(logging.WARNING):
        if operation == "record":
            assert (
                await preference.record_block_event(
                    db, user_id, course_id, "view", "notes"
                )
                is None
            )
        elif operation == "onboarding":
            assert await preference._load_onboarding_boosts(db, user_id) == {}
        else:
            assert (
                await preference.compute_block_preferences(db, user_id, course_id) == {}
            )
    assert str(user_id) in caplog.text
    assert "database unavailable" in caplog.text
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.asyncio
async def test_preference_programming_errors_propagate():
    db = SimpleNamespace(execute=AsyncMock(side_effect=AttributeError("bad query")))
    with pytest.raises(AttributeError, match="bad query"):
        await preference._load_onboarding_boosts(db, uuid.uuid4())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fetch", [automation.fetch_with_httpx, automation.fetch_with_browser]
)
async def test_invalid_url_is_logged_without_network_access(fetch, monkeypatch, caplog):
    def reject(_):
        raise ValidationError("blocked URL")

    monkeypatch.setattr("libs.url_validation.validate_url", reject)
    with caplog.at_level(logging.WARNING):
        assert await fetch("https://example.com") is None
    assert "URL validation failed" in caplog.text


@pytest.mark.asyncio
async def test_url_validator_programming_error_propagates(monkeypatch):
    def broken(_):
        raise AttributeError("validator bug")

    monkeypatch.setattr("libs.url_validation.validate_url", broken)
    with pytest.raises(AttributeError, match="validator bug"):
        await automation.fetch_with_httpx("https://example.com")


@pytest.mark.asyncio
async def test_browser_catches_playwright_error_but_not_programming_error(
    monkeypatch, caplog
):
    class PlaywrightError(Exception):
        pass

    module = types.ModuleType("playwright.async_api")
    module.Error = PlaywrightError
    module.async_playwright = lambda: (_ for _ in ()).throw(
        PlaywrightError("browser closed")
    )
    package = types.ModuleType("playwright")
    package.async_api = module
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.async_api", module)
    monkeypatch.setattr("libs.url_validation.validate_url", lambda url: url)
    with caplog.at_level(logging.ERROR):
        assert await automation.fetch_with_browser("https://example.com") is None
    assert "browser closed" in caplog.text
    module.async_playwright = lambda: (_ for _ in ()).throw(
        AttributeError("browser bug")
    )
    with pytest.raises(AttributeError, match="browser bug"):
        await automation.fetch_with_browser("https://example.com")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("offline"),
        json.JSONDecodeError("invalid response", "{", 1),
    ],
)
async def test_ollama_expected_errors_preserve_service_error(
    error, monkeypatch, caplog
):
    monkeypatch.setattr(
        preferences_llm, "get_ollama_models", AsyncMock(side_effect=error)
    )
    with caplog.at_level(logging.ERROR), pytest.raises(ExternalServiceError) as caught:
        await preferences_llm.list_ollama_models(base_url=None, user=SimpleNamespace())
    assert caught.value.__cause__ is error
    assert "Failed to fetch Ollama models" in caplog.text


@pytest.mark.asyncio
async def test_ollama_programming_errors_propagate(monkeypatch):
    monkeypatch.setattr(
        preferences_llm,
        "get_ollama_models",
        AsyncMock(side_effect=AttributeError("response bug")),
    )
    with pytest.raises(AttributeError, match="response bug"):
        await preferences_llm.list_ollama_models(base_url=None, user=SimpleNamespace())


def test_json_fallback_logs_diagnostics_without_response_content(caplog):
    with caplog.at_level(logging.DEBUG):
        assert parse_llm_json("private student response {broken}", default=[]) == []
    assert "parse failed" in caplog.text
    assert "private student response" not in caplog.text


@pytest.mark.parametrize("error", [BadZipFile("bad archive"), SyntaxError("bad XML")])
def test_corrupt_xlsx_is_logged_and_returns_empty_content(error, monkeypatch, caplog):
    module = types.ModuleType("openpyxl")

    def fail(*args, **kwargs):
        raise error

    module.load_workbook = fail
    monkeypatch.setitem(sys.modules, "openpyxl", module)
    with caplog.at_level(logging.WARNING):
        assert _extract_xlsx_fallback("lecture.xlsx") == ("lecture", "")
    assert "Invalid XLSX document lecture.xlsx" in caplog.text
    assert any(record.exc_info for record in caplog.records)
