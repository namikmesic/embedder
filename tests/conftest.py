from __future__ import annotations

from uuid import uuid4

import pytest

from config import load_config
from knowledgebase import KnowledgebaseManager
from server import VectorDBService


@pytest.fixture(scope="session")
async def manager() -> KnowledgebaseManager:
    """Shared KnowledgebaseManager for the test session."""
    config = load_config()
    mgr = KnowledgebaseManager(
        store_config=config.store,
        object_store_config=config.object_store,
        embedder_config=config.embedder,
        kb_defaults=config.kb_defaults,
    )
    await mgr.initialize()
    yield mgr
    await mgr.close()


@pytest.fixture(scope="session")
async def service(manager: KnowledgebaseManager) -> VectorDBService:
    return VectorDBService(manager)


@pytest.fixture(scope="session")
async def hybrid_service(manager: KnowledgebaseManager) -> VectorDBService:
    """A service whose default KB is looked up normally,
    but we create a dedicated hybrid KB for hybrid search tests."""
    # Create a KB with hybrid_alpha=0.5 for hybrid search testing.
    # Use a unique name so it doesn't collide across test runs.
    kb_name = f"hybrid-test-{uuid4().hex[:8]}"
    await manager.create_kb(name=kb_name, hybrid_alpha=0.5)
    svc = VectorDBService(manager)
    # Stash the KB name so tests can use it
    svc._hybrid_kb_name = kb_name  # type: ignore[attr-defined]
    yield svc
    # Cleanup: drop the table and registry row
    pool = manager._require_pool()
    record_row = await pool.fetchrow(
        "SELECT table_name FROM knowledgebases WHERE name = $1", kb_name
    )
    if record_row:
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {record_row['table_name']} CASCADE")
            await conn.execute("DELETE FROM knowledgebases WHERE name = $1", kb_name)
    manager._cache.pop(kb_name, None)


@pytest.fixture()
def tenant_id() -> str:
    return str(uuid4())
