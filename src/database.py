import os
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base


RAW_DATABASE_URL = os.environ["DATABASE_URL"]


def _to_asyncpg_url(url: str) -> str:
    """
    Convert a standard `postgresql://...?sslmode=require&channel_binding=require`
    URL (what Neon issues) into one asyncpg/SQLAlchemy can actually use.

    - swaps the driver to +asyncpg
    - drops the query string entirely: `sslmode` and `channel_binding` are
      libpq-only params. asyncpg's connect() doesn't accept them — passing
      them through raises `TypeError: connect() got an unexpected keyword
      argument 'sslmode'`. SSL is configured via connect_args instead (below).
    """
    parsed = urlparse(url)
    return urlunparse(("postgresql+asyncpg", parsed.netloc, parsed.path, parsed.params, "", parsed.fragment))


ASYNC_DATABASE_URL = _to_asyncpg_url(RAW_DATABASE_URL)

# Neon's `-pooler` hostname means PgBouncer in transaction-pooling mode.
# asyncpg prepares statements server-side and caches them per logical
# connection by default. Under transaction pooling, the physical backend
# connection can change between statements in the same session, so a cached
# prepared statement can reference a backend that no longer has it ->
# intermittent "prepared statement ... does not exist" errors under load.
# statement_cache_size=0 disables that cache. Cost: every query re-prepares
# server-side (small latency hit) — the correct tradeoff for a pooled conn.
_connect_args = {}
if "sslmode=require" in RAW_DATABASE_URL:
    _connect_args["ssl"] = True
if "-pooler" in RAW_DATABASE_URL:
    _connect_args["statement_cache_size"] = 0

engine = create_async_engine(ASYNC_DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)

# expire_on_commit=False: with a sync Session, accessing an attribute after
# commit() triggers an implicit lazy-load (a blocking DB round trip) to
# refresh it. With AsyncSession that same implicit lazy-load raises
# MissingGreenlet, because SQLAlchemy can't silently go async mid-attribute-
# access. Setting expire_on_commit=False keeps already-loaded attributes
# usable after commit without a refresh; anything you need fresh after
# commit must be re-queried explicitly.
AsyncSessionLocal = async_sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

Base = declarative_base()


async def get_db():
    """FastAPI dependency — yields an async session, closes it after the request."""
    async with AsyncSessionLocal() as db:
        yield db