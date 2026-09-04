import asyncio


async def sync_iter_to_async(sync_iterable):
    """
    Bridges a blocking, synchronous iterator (e.g. boto3's bedrock-runtime
    EventStream) into an async generator that yields items as they actually
    arrive, without blocking the event loop.

    Why this exists: boto3 has no async API. Naively doing
    `await asyncio.to_thread(list, sync_iterable)` would work, but it drains
    the ENTIRE iterator in the background thread before returning anything --
    defeating streaming, since nothing reaches the caller until every chunk
    has already arrived. This runs the blocking iteration in a background
    thread and pushes each item onto an asyncio.Queue via
    loop.call_soon_threadsafe as it's produced, so the async generator can
    yield incrementally, in near-real-time, while the event loop stays free
    for other concurrent work.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _producer():
        try:
            for item in sync_iterable:
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    loop.run_in_executor(None, _producer)

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item