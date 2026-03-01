from __future__ import annotations

from uuid import uuid4

import pytest

from server import VectorDBService


async def test_store_then_search(service: VectorDBService, tenant_id: str):
    result = await service.store_chunk(
        text="PostgreSQL supports row-level security policies.",
        source_id="doc-1",
        tenant_id=tenant_id,
        content_date="2025-01-15",
        title="PG Docs",
    )
    assert result["status"] == "created"
    chunk_id = result["chunk_id"]

    results = await service.search("row level security", tenant_id)
    assert len(results) >= 1
    hit = results[0]
    assert hit["chunk_id"] == chunk_id
    assert hit["title"] == "PG Docs"
    assert hit["text"] == "PostgreSQL supports row-level security policies."
    assert 0 < hit["score"] <= 1.0


async def test_idempotent_store(service: VectorDBService, tenant_id: str):
    text = "Idempotent chunk content for deduplication test."
    r1 = await service.store_chunk(
        text=text, source_id="idem-1", tenant_id=tenant_id, content_date="2025-03-01",
    )
    assert r1["status"] == "created"

    r2 = await service.store_chunk(
        text=text, source_id="idem-1", tenant_id=tenant_id, content_date="2025-03-01",
    )
    assert r2["status"] == "unchanged"
    assert r2["chunk_id"] == r1["chunk_id"]


async def test_update_chunk(service: VectorDBService, tenant_id: str):
    r1 = await service.store_chunk(
        text="Version one of the document.",
        source_id="update-1",
        tenant_id=tenant_id,
        content_date="2025-02-01",
    )
    assert r1["status"] == "created"

    r2 = await service.store_chunk(
        text="Version two with different content.",
        source_id="update-1",
        tenant_id=tenant_id,
        content_date="2025-02-01",
    )
    assert r2["status"] == "updated"
    assert r2["chunk_id"] != r1["chunk_id"]

    results = await service.search("Version two", tenant_id)
    assert any(r["chunk_id"] == r2["chunk_id"] for r in results)
    assert not any(r["chunk_id"] == r1["chunk_id"] for r in results)

    count = await service.count_chunks(tenant_id)
    assert count["total_chunks"] == 1


async def test_reject_oversized(service: VectorDBService, tenant_id: str):
    _, _, embedder = await service.manager.get_kb("default")
    huge = "word " * (embedder.max_tokens + 100)
    with pytest.raises(ValueError, match="Chunk too large"):
        await service.store_chunk(
            text=huge, source_id="big", tenant_id=tenant_id, content_date="2025-01-01",
        )


async def test_hybrid_search(hybrid_service: VectorDBService, tenant_id: str):
    kb_name = hybrid_service._hybrid_kb_name  # type: ignore[attr-defined]
    await hybrid_service.store_chunk(
        text="The pangolin is a scaly mammal found in Asia and Africa.",
        source_id="hybrid-1",
        tenant_id=tenant_id,
        content_date="2025-04-01",
        knowledgebase=kb_name,
    )
    await hybrid_service.store_chunk(
        text="Machine learning models require large training datasets.",
        source_id="hybrid-2",
        tenant_id=tenant_id,
        content_date="2025-04-01",
        knowledgebase=kb_name,
    )

    results = await hybrid_service.search(
        "pangolin scaly mammal", tenant_id, knowledgebase=kb_name
    )
    assert len(results) >= 1
    assert results[0]["source_id"] == "hybrid-1"


async def test_time_range_filter(service: VectorDBService, tenant_id: str):
    for month, sid in [("01", "jan"), ("03", "mar"), ("06", "jun")]:
        await service.store_chunk(
            text=f"Event in month {month} of the year.",
            source_id=sid,
            tenant_id=tenant_id,
            content_date=f"2025-{month}-15",
        )

    results = await service.search(
        "event", tenant_id, after="2025-02-01", before="2025-05-01",
    )
    source_ids = {r["source_id"] for r in results}
    assert "mar" in source_ids
    assert "jan" not in source_ids
    assert "jun" not in source_ids


async def test_search_empty(service: VectorDBService, tenant_id: str):
    results = await service.search("anything", tenant_id)
    assert results == []


async def test_top_k_exceeded(service: VectorDBService, tenant_id: str):
    with pytest.raises(ValueError, match="top_k must be <= 200"):
        await service.search("query", tenant_id, top_k=201)


async def test_delete_and_count(service: VectorDBService, tenant_id: str):
    await service.store_chunk(
        text="First chunk to delete.",
        source_id="del-1",
        tenant_id=tenant_id,
        content_date="2025-01-01",
    )
    await service.store_chunk(
        text="Second chunk to keep.",
        source_id="del-2",
        tenant_id=tenant_id,
        content_date="2025-01-01",
    )
    assert (await service.count_chunks(tenant_id))["total_chunks"] == 2

    deleted = await service.delete_chunks("del-1", tenant_id)
    assert deleted["chunks_deleted"] == 1
    assert (await service.count_chunks(tenant_id))["total_chunks"] == 1

    deleted = await service.delete_chunks("nonexistent", tenant_id)
    assert deleted["chunks_deleted"] == 0


async def test_tenant_isolation(service: VectorDBService):
    tenant_a = str(uuid4())
    tenant_b = str(uuid4())

    await service.store_chunk(
        text="Secret data for tenant A only.",
        source_id="iso-1",
        tenant_id=tenant_a,
        content_date="2025-01-01",
    )

    results_b = await service.search("secret data", tenant_b)
    assert results_b == []

    results_a = await service.search("secret data", tenant_a)
    assert len(results_a) == 1
    assert results_a[0]["source_id"] == "iso-1"
