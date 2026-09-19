"""Company name lookup tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_settings
from core.ingest.company_names import (
    STATIC_NAMES,
    get_company_name,
    get_company_names,
)


def test_static_names_cover_default_universe():
    settings = load_settings()
    names = get_company_names(settings["universe"], allow_network=False)
    missing = [t for t, n in names.items() if n == t]
    assert not missing, f"no name for: {missing}"


def test_offline_lookup_uses_static_map():
    names = get_company_names(["AAPL", "MSFT", "DE"], allow_network=False)
    assert names["AAPL"] == "Apple Inc."
    assert names["MSFT"] == "Microsoft Corporation"
    assert names["DE"] == "Deere & Company"


def test_unknown_ticker_falls_back_to_ticker():
    names = get_company_names(["ZZZZNOTREAL"], allow_network=False)
    assert names["ZZZZNOTREAL"] == "ZZZZNOTREAL"


def test_get_company_name_single():
    assert get_company_name("NVDA", allow_network=False) == "NVIDIA Corporation"


def test_duplicate_tickers_are_stable():
    names = get_company_names(["AAPL", "AAPL"], allow_network=False)
    assert list(names.keys()) == ["AAPL"]


def test_block_xyz_naming():
    # Block renamed SQ -> XYZ; both should resolve to the company name.
    assert STATIC_NAMES["XYZ"] == STATIC_NAMES["SQ"] == "Block, Inc."
