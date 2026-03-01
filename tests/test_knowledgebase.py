from __future__ import annotations

from uuid import uuid4

import pytest

from knowledgebase import KnowledgebaseManager
from server import VectorDBService


@pytest.fixture()
async def test_kb(manager: KnowledgebaseManager):
    """Create a uniquely-named KB for a single test, clean up on teardown."""
    kb_name = f"test-kb-{uuid4().hex[:8]}"
    record = await manager.create_kb(name=kb_name)
    yield record
    # Teardown: drop table + registry row
    pool = manager._require_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {record.table_name} CASCADE")
        await conn.execute("DELETE FROM knowledgebases WHERE name = $1", kb_name)
    manager._cache.pop(kb_name, None)


async def test_create_knowledgebase(manager: KnowledgebaseManager, test_kb):
    """Verify that creating a KB produces a registry entry and table."""
    pool = manager._require_pool()
    async with pool.acquire() as conn:
        # Registry row exists
        row = await conn.fetchrow(
            "SELECT * FROM knowledgebases WHERE name = $1", test_kb.name
        )
        assert row is not None
        assert row["dimension"] > 0

        # Table exists
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
            test_kb.table_name,
        )
        assert exists


async def test_create_duplicate_name(manager: KnowledgebaseManager, test_kb):
    """Creating a KB with the same name should raise ValueError."""
    with pytest.raises(ValueError, match="already exists"):
        await manager.create_kb(name=test_kb.name)


async def test_store_and_search_custom_kb(
    service: VectorDBService, manager: KnowledgebaseManager, test_kb, tenant_id: str
):
    """Round-trip store + search in a non-default KB."""
    result = await service.store_chunk(
        text="Custom KB content for testing.",
        source_id="custom-1",
        tenant_id=tenant_id,
        content_date="2025-06-01",
        knowledgebase=test_kb.name,
    )
    assert result["status"] == "created"

    results = await service.search(
        "custom KB content", tenant_id, knowledgebase=test_kb.name
    )
    assert len(results) >= 1
    assert results[0]["text"] == "Custom KB content for testing."


async def test_kb_isolation(
    service: VectorDBService, manager: KnowledgebaseManager, test_kb, tenant_id: str
):
    """Same tenant, different KBs — data doesn't leak across KBs."""
    # Store in default KB
    await service.store_chunk(
        text="Data in the default knowledgebase.",
        source_id="iso-default",
        tenant_id=tenant_id,
        content_date="2025-06-01",
    )

    # Store in custom KB
    await service.store_chunk(
        text="Data in the custom knowledgebase.",
        source_id="iso-custom",
        tenant_id=tenant_id,
        content_date="2025-06-01",
        knowledgebase=test_kb.name,
    )

    # Search in default — should only find default data
    default_results = await service.search("knowledgebase data", tenant_id)
    default_sources = {r["source_id"] for r in default_results}
    assert "iso-default" in default_sources
    assert "iso-custom" not in default_sources

    # Search in custom — should only find custom data
    custom_results = await service.search(
        "knowledgebase data", tenant_id, knowledgebase=test_kb.name
    )
    custom_sources = {r["source_id"] for r in custom_results}
    assert "iso-custom" in custom_sources
    assert "iso-default" not in custom_sources


async def test_default_kb_backward_compat(manager: KnowledgebaseManager):
    """The default KB should use the 'chunks' table."""
    record, store, embedder = await manager.get_kb("default")
    assert record.name == "default"
    assert record.table_name == "chunks"
    assert record.dimension == 768


async def test_list_knowledgebases(
    service: VectorDBService, manager: KnowledgebaseManager, test_kb
):
    """list_knowledgebases should return at least the default and test KBs."""
    kbs = await service.list_knowledgebases()
    names = {kb["name"] for kb in kbs}
    assert "default" in names
    assert test_kb.name in names


async def test_embedder_cache_sharing(manager: KnowledgebaseManager, test_kb):
    """Two KBs with the same model should share the same Embedder instance."""
    _, _, default_emb = await manager.get_kb("default")
    _, _, test_emb = await manager.get_kb(test_kb.name)
    # Both use the default BAAI/bge-base-en-v1.5 model
    assert default_emb is test_emb
