import os
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import event
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

# ivfflat.probes: pgvector's own default is 1, i.e. a similarity query only
# scans one of the ix_chunks_embedding index's `lists=100` partitions. With
# only 236 rows in `chunks` right now, that's ~2.4 rows per partition --
# severely over-partitioned for this table size -- so a real query's true
# nearest neighbors frequently land in a partition that never gets probed.
# Verified directly (not assumed), across all 38 real questions in
# Golden_questions.json at probes=1: 5/38 got fewer than the requested k=3
# rows back, 4 of those got ZERO rows -- meaning generate.py's
# SIMILARITY_THRESHOLD guard would fire NO_CONTEXT_ANSWER regardless of
# whether the real answer existed in the DB. At probes=10: 0/38 truncated.
# asyncpg's `server_settings` connect kwarg does NOT work for this --
# extension-defined GUCs aren't recognized in the startup packet before the
# extension loads (raises `UndefinedObjectError`, verified). Setting it via
# a real `SET` after connect, through the DBAPI-level "connect" pool event,
# is the asyncpg-specific way to run this on every new physical connection;
# `dbapi_connection.run_async(...)` is SQLAlchemy's asyncpg adapter's own
# escape hatch for exactly this (a sync event callback wrapping an async
# asyncpg call). 10 is not benchmarked against a larger/labeled dataset --
# same "starting value, revisit later" caveat as SIMILARITY_THRESHOLD and
# `lists=100` itself, which remains oversized for the current row count and
# should be revisited (smaller `lists`, in a new migration) as the table
# grows past a few thousand rows.
IVFFLAT_PROBES = int(os.environ.get("IVFFLAT_PROBES", "10"))


async def _try_set_ivfflat_probes(asyncpg_conn):
    try:
        await asyncpg_conn.execute(f"SET ivfflat.probes = {IVFFLAT_PROBES}")
    except Exception:
        # `ivfflat.probes` is a GUC the `vector` extension defines -- on a
        # brand-new database where `CREATE EXTENSION vector` hasn't run yet,
        # this connection is the one about to run it (tests/conftest.py's
        # clean_db fixture does exactly this via this same engine). Failing
        # here would raise out of the "connect" event and break that very
        # first connection -- verified: this is what broke CI (a fresh
        # pgvector/pgvector:pg16 container, no migration run, extension
        # created by the test fixture itself). Silently skip; the fixture's
        # own CREATE EXTENSION runs right after, and every later connection
        # in the pool succeeds once the extension exists.
        pass


@event.listens_for(engine.sync_engine, "connect")
def _set_ivfflat_probes(dbapi_connection, connection_record):
    dbapi_connection.run_async(_try_set_ivfflat_probes)

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