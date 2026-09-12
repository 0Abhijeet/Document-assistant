from langfuse import get_client


def get_langfuse_client():
    """Thin wrapper for consistency with get_embeddings()/get_bedrock_client()
    elsewhere in this codebase. get_client() reads LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY, and LANGFUSE_HOST from the environment automatically,
    and degrades gracefully -- logs a warning, becomes a safe no-op -- if
    they're missing. Verified directly (not assumed): calling
    start_as_current_observation() with no keys set raises nothing. Safe to
    call in any environment, including ones with Langfuse unconfigured
    (local dev without keys, CI, the pytest suite).
    """
    return get_client()