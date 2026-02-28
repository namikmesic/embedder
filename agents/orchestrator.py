from __future__ import annotations

import json
import logging
from typing import Callable, Optional

import anyio

from agents.base import Message, clean_json
from agents.budget import ContextBudgetManager, FileTokenInfo, Partition
from agents.factory import create_agent
from agents.prompts import SYNTHESIZER_SYSTEM
from agents.tasks import run_explorer, run_followup, run_scanner
from config.models import AgentConfig
from domain.knowledge import (
    BootstrapMap,
    BootstrapStatus,
    Connection,
    Entity,
    Finding,
    KnowledgeDoc,
    Unknown,
)
from sources.base import Source

logger = logging.getLogger(__name__)


class BootstrapOrchestrator:
    def __init__(
        self,
        config: AgentConfig,
        source: Source,
        repo_id: str,
        repo_url: str,
        file_globs: list[str],
        exclude_patterns: list[str],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self._config = config
        self._source = source
        self._repo_id = repo_id
        self._repo_url = repo_url
        self._file_globs = file_globs
        self._exclude_patterns = exclude_patterns
        self._progress = progress_callback

        self._orchestrator_agent = create_agent(config.orchestrator)
        self._sub_agent_config = config.sub_agent
        self._budget_mgr = ContextBudgetManager(
            context_window=config.sub_agent.context_window,
            budget_fraction=config.context_budget_fraction,
        )

        self._total_tokens = 0

    def _report(self, fraction: float, message: str) -> None:
        if self._progress:
            self._progress(fraction, message)

    async def run(self) -> tuple[BootstrapMap, list[KnowledgeDoc]]:
        bmap = BootstrapMap(repo_id=self._repo_id, repo_url=self._repo_url, status=BootstrapStatus.SCANNING)

        try:
            # Scan
            self._report(0.05, "Scanning file tree...")
            file_tree = self._source.get_file_tree(self._file_globs, self._exclude_patterns)
            raw_partitions = await run_scanner(create_agent(self._sub_agent_config), file_tree, self._source.get_info())

            self._report(0.10, "Building partitions...")
            partitions = self._build_partitions(raw_partitions)

            # Explore
            bmap.status = BootstrapStatus.EXPLORING
            self._report(0.15, f"Exploring {len(partitions)} partitions...")
            findings = await self._run_explorers(partitions)

            all_entities, all_connections, all_unknowns = self._merge_findings(findings)
            self._report(0.50, f"Found {len(all_entities)} entities, {len(all_connections)} connections")

            # Follow-up rounds for high-priority unknowns
            rounds = 0
            for round_num in range(self._config.max_followup_rounds):
                priority_unknowns = [u for u in all_unknowns if u.priority <= 2]
                if not priority_unknowns:
                    break

                self._report(0.55 + round_num * 0.05, f"Follow-up round {round_num + 1}: {len(priority_unknowns)} unknowns...")
                followup_finding = await run_followup(
                    create_agent(self._sub_agent_config), priority_unknowns, self._source,
                    self._file_globs, self._exclude_patterns,
                )
                self._total_tokens += followup_finding.tokens_consumed

                for ent in followup_finding.entities:
                    if not any(e.name == ent.name for e in all_entities):
                        all_entities.append(ent)
                all_connections.extend(followup_finding.connections)
                all_unknowns = followup_finding.unknowns
                rounds += 1

            bmap.entities = all_entities
            bmap.connections = all_connections
            bmap.unresolved_unknowns = all_unknowns
            bmap.agent_rounds = 1 + rounds
            bmap.total_tokens_consumed = self._total_tokens

            # Generate docs
            bmap.status = BootstrapStatus.GENERATING_DOCS
            self._report(0.70, "Generating documentation...")
            docs = await self._generate_docs(bmap)

            bmap.status = BootstrapStatus.COMPLETE
            self._report(0.90, f"Generated {len(docs)} documentation pages")
            return bmap, docs

        except Exception:
            bmap.status = BootstrapStatus.ERROR
            logger.exception("Bootstrap failed for %s", self._repo_id)
            raise

    def _build_partitions(self, raw_partitions: list[dict]) -> list[Partition]:
        all_files = self._source.discover_files(self._file_globs, self._exclude_patterns)
        file_map: dict[str, Path] = {
            str(f.relative_to(self._source.local_path)): f for f in all_files
        }

        assigned_files: set[str] = set()
        partitions: list[Partition] = []

        for raw in raw_partitions:
            name = raw.get("name", "unknown")
            description = raw.get("description", "")
            files: list[FileTokenInfo] = []

            for rel_path in raw.get("files", []):
                abs_path = file_map.get(rel_path)
                if abs_path and rel_path not in assigned_files:
                    content = self._source.get_file_content(abs_path)
                    finfo = self._budget_mgr.compute_file_tokens(abs_path, content, self._source.local_path)
                    files.append(finfo)
                    assigned_files.add(rel_path)

            if files:
                partitions.append(Partition(name=name, description=description, files=files))

        # Catch unassigned files
        unassigned = [
            rel for rel in file_map if rel not in assigned_files
        ]
        if unassigned:
            logger.warning("%d files not assigned by scanner, creating 'unassigned' partition", len(unassigned))
            files = []
            for rel in unassigned:
                abs_path = file_map[rel]
                content = self._source.get_file_content(abs_path)
                finfo = self._budget_mgr.compute_file_tokens(abs_path, content, self._source.local_path)
                files.append(finfo)
            partitions.append(Partition(name="unassigned", description="Files not assigned by scanner", files=files))

        # Split oversized partitions
        final_partitions: list[Partition] = []
        for p in partitions:
            final_partitions.extend(self._budget_mgr.split_partition_to_budget(p))

        logger.info("Built %d partitions from %d scanner suggestions", len(final_partitions), len(raw_partitions))
        return final_partitions

    async def _run_explorers(self, partitions: list[Partition]) -> list[Finding]:
        findings: list[Finding] = []
        limiter = anyio.CapacityLimiter(self._config.max_concurrent_agents)
        completed = 0
        total = len(partitions)

        async def explore_one(partition: Partition) -> None:
            nonlocal completed
            async with limiter:
                agent = create_agent(self._sub_agent_config)
                finding = await run_explorer(agent, partition, self._source)
                findings.append(finding)
                self._total_tokens += finding.tokens_consumed
                completed += 1
                progress = 0.15 + (completed / total) * 0.35
                self._report(progress, f"Explored {completed}/{total}: {partition.name}")

        async with anyio.create_task_group() as tg:
            for partition in partitions:
                tg.start_soon(explore_one, partition)

        return findings

    def _merge_findings(self, findings: list[Finding]) -> tuple[list[Entity], list[Connection], list[Unknown]]:
        seen: set[str] = set()
        all_entities: list[Entity] = []
        all_unknowns: list[Unknown] = []
        name_to_id: dict[str, str] = {}

        for finding in findings:
            for ent in finding.entities:
                if ent.name not in seen:
                    seen.add(ent.name)
                    all_entities.append(ent)
                    name_to_id[ent.name] = ent.entity_id
            all_unknowns.extend(finding.unknowns)

        all_connections: list[Connection] = []
        for finding in findings:
            for conn in finding.connections:
                all_connections.append(Connection(
                    source_entity_id=name_to_id.get(conn.source_entity_id, conn.source_entity_id),
                    target_entity_id=name_to_id.get(conn.target_entity_id, conn.target_entity_id),
                    kind=conn.kind,
                    description=conn.description,
                    evidence_file=conn.evidence_file,
                ))

        return all_entities, all_connections, all_unknowns

    async def _generate_docs(self, bootstrap_map: BootstrapMap) -> list[KnowledgeDoc]:
        map_summary = {
            "repo_id": bootstrap_map.repo_id,
            "repo_url": bootstrap_map.repo_url,
            "entities": [
                {
                    "name": e.name,
                    "kind": e.kind.value,
                    "file_path": e.file_path,
                    "description": e.description,
                    "signature": e.signature,
                    "tags": e.tags,
                }
                for e in bootstrap_map.entities
            ],
            "connections": [
                {
                    "source": e.source_entity_id,
                    "target": e.target_entity_id,
                    "kind": e.kind.value,
                    "description": e.description,
                }
                for e in bootstrap_map.connections
            ],
        }

        user_content = (
            f"Generate comprehensive documentation for this project.\n\n"
            f"Knowledge map:\n{json.dumps(map_summary, indent=2)}"
        )

        result = await self._orchestrator_agent.complete(
            system_prompt=SYNTHESIZER_SYSTEM,
            messages=[Message(role="user", content=user_content)],
            response_format={"type": "json_object"},
        )
        self._total_tokens += result.tokens_used

        parsed = json.loads(clean_json(result.content))

        docs: list[KnowledgeDoc] = []
        for d in parsed.get("docs", []):
            docs.append(KnowledgeDoc(
                title=d.get("title", "Untitled"),
                content=d.get("content", ""),
                doc_type=d.get("doc_type", "overview"),
                source_entities=d.get("source_entities", []),
                repo_id=self._repo_id,
                repo_url=self._repo_url,
            ))

        return docs
