"""Ingestion subpackage."""
from core.ingest.company_names import get_company_name, get_company_names
from core.ingest.prices import fetch_prices, last_trading_day, synthetic_prices

__all__ = [
    "fetch_prices",
    "synthetic_prices",
    "last_trading_day",
    "get_company_names",
    "get_company_name",
]
