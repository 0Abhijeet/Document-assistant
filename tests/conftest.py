import pytest


class FakeEmbeddings:
    """Deterministic fake embeddings — avoids downloading the real HF model in CI/tests."""

    def embed_documents(self, texts):
        return [self._fake_vector(t) for t in texts]

    def embed_query(self, text):
        return self._fake_vector(text)

    @staticmethod
    def _fake_vector(text):
        seed = sum(ord(c) for c in text) % 1000
        return [((seed + i) % 100) / 100.0 for i in range(384)]


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Applied to every test — replaces the real embedding model with the fake one.
    Unchanged from the sync version: monkeypatching a plain attribute needs no
    async handling regardless of what the code underneath is."""
    from src import ingest

    fake = FakeEmbeddings()
    monkeypatch.setattr(ingest, "_embeddings", fake)
    monkeypatch.setattr(ingest, "get_embeddings", lambda: fake)
    yield fake