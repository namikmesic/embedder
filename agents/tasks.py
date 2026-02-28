from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.base import LLMAgent, Message, clean_json
from agents.budget import Partition
from agents.prompts import EXPLORER_SYSTEM, FOLLOWUP_SYSTEM, SCANNER_SYSTEM
from domain.knowledge import (
    Connection,
    ConnectionKind,
    Entity,
    EntityKind,
    Finding,
    Unknown,
)
from sources.base import Source

logger = logging.getLogger(__name__)


async def run_scanner(agent: LLMAgent, file_tree: str, source_info: dict) -> list[dict]:
    user_content = f"Project info:\n{json.dumps(source_info, indent=2)}\n\nFile tree:\n{file_tree}"

    result = await agent.complete(
        system_prompt=SCANNER_SYSTEM,
        messages=[Message(role="user", content=user_content)],
        response_format={"type": "json_object"},
    )

    parsed = json.loads(clean_json(result.content))
    partitions = parsed.get("partitions", [])
    logger.info("Scanner suggested %d partitions (tokens: %d)", len(partitions), result.tokens_used)
    return partitions


async def run_explorer(agent: LLMAgent, partition: Partition, source: Source) -> Finding:
    file_blocks = []
    for finfo in partition.files:
        content = source.get_file_content(finfo.path)
        file_blocks.append(f"=== FILE: {finfo.relative_path} ===\n{content}\n=== END FILE ===")

    user_content = (
        f"Partition: {partition.name}\n"
        f"Description: {partition.description}\n"
        f"Files ({len(partition.files)}, ~{partition.total_tokens} tokens):\n\n"
        + "\n\n".join(file_blocks)
    )

    result = await agent.complete(
        system_prompt=EXPLORER_SYSTEM,
        messages=[Message(role="user", content=user_content)],
        response_format={"type": "json_object"},
    )

    finding = _parse_finding(json.loads(clean_json(result.content)), result.tokens_used)
    logger.info(
        "Explorer '%s': %d entities, %d connections, %d unknowns",
        partition.name, len(finding.entities), len(finding.connections), len(finding.unknowns),
    )
    return finding


async def run_followup(
    agent: LLMAgent,
    unknowns: list[Unknown],
    source: Source,
    globs: list[str],
    excludes: list[str],
) -> Finding:
    file_paths = {u.file_path for u in unknowns if u.file_path}

    all_files = source.discover_files(globs, excludes)
    relevant = [f for f in all_files if str(f.relative_to(source.local_path)) in file_paths]

    file_blocks = []
    for fpath in relevant[:10]:  # Budget guard
        rel = str(fpath.relative_to(source.local_path))
        file_blocks.append(f"=== FILE: {rel} ===\n{source.get_file_content(fpath)}\n=== END FILE ===")

    unknowns_desc = "\n".join(
        f"- [{u.priority}] {u.description} (file: {u.file_path or 'unknown'})"
        for u in unknowns
    )
    user_content = f"Unknowns to investigate:\n{unknowns_desc}\n\nRelevant source files:\n\n" + "\n\n".join(file_blocks)

    result = await agent.complete(
        system_prompt=FOLLOWUP_SYSTEM,
        messages=[Message(role="user", content=user_content)],
        response_format={"type": "json_object"},
    )

    finding = _parse_finding(json.loads(clean_json(result.content)), result.tokens_used)
    logger.info("Follow-up: %d entities found, %d unknowns remain", len(finding.entities), len(finding.unknowns))
    return finding


def _parse_finding(data: dict, tokens_used: int) -> Finding:
    entities = []
    for e in data.get("entities", []):
        try:
            kind = EntityKind(e.get("kind", "other"))
        except ValueError:
            kind = EntityKind.OTHER
        entities.append(Entity(
            name=e.get("name", ""),
            kind=kind,
            file_path=e.get("file_path", ""),
            line_range=e.get("line_range"),
            description=e.get("description", ""),
            signature=e.get("signature"),
            tags=e.get("tags", []),
        ))

    connections = []
    for c in data.get("connections", []):
        try:
            kind = ConnectionKind(c.get("kind", "depends_on"))
        except ValueError:
            kind = ConnectionKind.DEPENDS_ON
        connections.append(Connection(
            source_entity_id=c.get("source", ""),
            target_entity_id=c.get("target", ""),
            kind=kind,
            description=c.get("description", ""),
            evidence_file=c.get("evidence_file"),
        ))

    unknowns = [
        Unknown(
            description=u.get("description", ""),
            file_path=u.get("file_path"),
            related_entity_ids=u.get("related_entities", []),
            priority=u.get("priority", 3),
        )
        for u in data.get("unknowns", [])
    ]

    return Finding(
        agent_scope=data.get("agent_scope", ""),
        files_examined=data.get("files_examined", []),
        tokens_consumed=tokens_used,
        entities=entities,
        connections=connections,
        unknowns=unknowns,
        summary=data.get("summary", ""),
    )
