#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import click

from app import AppContext, _init_app, _run, pass_app
from config.loader import load_config
from config.models import RepoConfig
from ingest.chunker import Chunker
from ingest.pipeline import ingest_repo


logger = logging.getLogger("git_rag")


def _progress_echo(fraction: float, message: str) -> None:
    pct = int(fraction * 100)
    click.echo(f"[{pct:3d}%] {message}", err=True)


def _json_output(data: Any) -> None:
    click.echo(json.dumps(data, indent=2, default=str))


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """git-rag — turn git repositories into queryable knowledge bases."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    config = load_config()
    app = _run(_init_app(config))
    ctx.obj = app

    ctx.call_on_close(lambda: _run(app.store.save()))


@cli.command()
@click.argument("url")
@click.option("-b", "--branch", default="main", show_default=True, help="Branch to track.")
@click.option("--file-globs", multiple=True, help="Glob patterns for files to index (repeatable).")
@click.option("--exclude", multiple=True, help="Glob patterns to exclude (repeatable).")
@click.option("--no-sync", is_flag=True, help="Register only — don't clone and ingest immediately.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@pass_app
def add(app: AppContext, url: str, branch: str, file_globs: tuple[str, ...], exclude: tuple[str, ...], no_sync: bool, as_json: bool) -> None:
    """Add a git repository as a knowledge base."""
    file_globs_list = list(file_globs) if file_globs else RepoConfig.model_fields["file_globs"].default_factory()
    exclude_list = list(exclude) if exclude else RepoConfig.model_fields["exclude_patterns"].default_factory()

    repo_config = RepoConfig(
        url=url,
        branch=branch,
        file_globs=file_globs_list,
        exclude_patterns=exclude_list,
    )

    repo_state = app.state.add_repo(repo_config)
    result: dict[str, Any] = {"repo_id": repo_config.repo_id, "url": url, "branch": branch, "status": "registered"}

    if not no_sync:
        callback = None if as_json else _progress_echo
        ingest_result = _run(ingest_repo(
            repo_state, app.config, app.embedder, app.store,
            app.preprocessor, app.chunker, app.state,
            force_full=True, progress_callback=callback,
        ))
        result.update(ingest_result)

    if as_json:
        _json_output(result)
    else:
        status = result.get("status", "registered")
        click.echo(f"Added repo {repo_config.repo_id} ({url}, branch: {branch})")
        if status == "ready":
            click.echo(f"  Files processed: {result.get('files_processed', 0)}")
            click.echo(f"  Chunks created:  {result.get('chunks_created', 0)}")
            click.echo(f"  Total chunks:    {result.get('total_chunks_in_store', 0)}")
        elif status == "error":
            click.echo(f"  Error: {result.get('error', 'unknown')}", err=True)
        elif no_sync:
            click.echo("  Skipped sync (use 'sync' to ingest later).")


@cli.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@pass_app
def list_repos(app: AppContext, as_json: bool) -> None:
    """List all registered repositories."""
    repos = []
    for rs in app.state.list_repos():
        repos.append({
            "repo_id": rs.config.repo_id,
            "url": rs.config.url,
            "branch": rs.config.branch,
            "status": rs.status,
            "chunk_count": rs.chunk_count,
            "last_ingested_commit": rs.last_ingested_commit,
            "last_ingested_at": rs.last_ingested_at,
            "error": rs.error,
        })

    if as_json:
        _json_output(repos)
        return

    if not repos:
        click.echo("No repositories registered. Use 'add' to register one.")
        return

    click.echo(f"{'REPO ID':<14} {'STATUS':<10} {'CHUNKS':>7}  {'URL'}")
    click.echo("-" * 70)
    for r in repos:
        click.echo(f"{r['repo_id']:<14} {r['status']:<10} {r['chunk_count']:>7}  {r['url']}")


@cli.command()
@click.argument("repo_id")
@click.option("--keep-chunks", is_flag=True, help="Don't delete indexed chunks from the vector store.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@pass_app
def remove(app: AppContext, repo_id: str, keep_chunks: bool, as_json: bool) -> None:
    """Remove a repository and its indexed chunks."""
    repo_state = app.state.get_repo(repo_id)
    if not repo_state:
        raise click.ClickException(f"Repository '{repo_id}' not found.")

    deleted_chunks = 0
    if not keep_chunks:
        deleted_chunks = _run(app.store.delete_by_metadata("repo_id", repo_id))
        _run(app.store.save())

    app.state.remove_repo(repo_id)

    result = {"repo_id": repo_id, "removed": True, "chunks_deleted": deleted_chunks}

    if as_json:
        _json_output(result)
    else:
        click.echo(f"Removed repo {repo_id}.")
        if not keep_chunks:
            click.echo(f"  Deleted {deleted_chunks} chunks from the vector store.")


@cli.command()
@click.argument("repo_id", required=False, default=None)
@click.option("--force-full", is_flag=True, help="Force full re-ingestion instead of incremental.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@pass_app
def sync(app: AppContext, repo_id: Optional[str], force_full: bool, as_json: bool) -> None:
    """Pull latest changes and re-ingest one or all repos."""
    if repo_id:
        repo_state = app.state.get_repo(repo_id)
        if not repo_state:
            raise click.ClickException(f"Repository '{repo_id}' not found.")
        repos_to_sync = [repo_state]
    else:
        repos_to_sync = app.state.list_repos()

    if not repos_to_sync:
        raise click.ClickException("No repositories to sync.")

    callback = None if as_json else _progress_echo
    results = []
    for rs in repos_to_sync:
        if not as_json:
            click.echo(f"Syncing {rs.config.repo_id} ({rs.config.url})...")
        result = _run(ingest_repo(
            rs, app.config, app.embedder, app.store,
            app.preprocessor, app.chunker, app.state,
            force_full=force_full, progress_callback=callback,
        ))
        results.append(result)

    if as_json:
        _json_output(results)
    else:
        for r in results:
            status = r.get("status", "?")
            if status == "up_to_date":
                click.echo(f"  {r['repo_id']}: already up to date.")
            elif status == "ready":
                click.echo(f"  {r['repo_id']}: {r.get('files_processed', 0)} files, {r.get('chunks_created', 0)} chunks.")
            elif status == "error":
                click.echo(f"  {r['repo_id']}: ERROR — {r.get('error', 'unknown')}", err=True)


@cli.command()
@click.argument("repo_id", required=False, default=None)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@pass_app
def status(app: AppContext, repo_id: Optional[str], as_json: bool) -> None:
    """Show stats and ingestion state for knowledge bases."""
    total_vectors = _run(app.store.count())

    if repo_id:
        repo_state = app.state.get_repo(repo_id)
        if not repo_state:
            raise click.ClickException(f"Repository '{repo_id}' not found.")

        data = {
            "repo_id": repo_state.config.repo_id,
            "url": repo_state.config.url,
            "branch": repo_state.config.branch,
            "status": repo_state.status,
            "chunk_count": repo_state.chunk_count,
            "last_ingested_commit": repo_state.last_ingested_commit,
            "last_ingested_at": repo_state.last_ingested_at,
            "error": repo_state.error,
            "file_globs": repo_state.config.file_globs,
            "exclude_patterns": repo_state.config.exclude_patterns,
            "total_vectors_in_store": total_vectors,
        }

        if as_json:
            _json_output(data)
        else:
            click.echo(f"Repo:    {data['repo_id']}")
            click.echo(f"URL:     {data['url']}")
            click.echo(f"Branch:  {data['branch']}")
            click.echo(f"Status:  {data['status']}")
            click.echo(f"Chunks:  {data['chunk_count']}")
            if data['last_ingested_commit']:
                click.echo(f"Commit:  {data['last_ingested_commit']}")
            if data['last_ingested_at']:
                click.echo(f"Synced:  {data['last_ingested_at']}")
            if data['error']:
                click.echo(f"Error:   {data['error']}")
            click.echo(f"Vectors: {data['total_vectors_in_store']} (total in store)")
        return

    repos = app.state.list_repos()
    data = {
        "total_repos": len(repos),
        "total_vectors": total_vectors,
        "repos": [
            {
                "repo_id": rs.config.repo_id,
                "url": rs.config.url,
                "status": rs.status,
                "chunk_count": rs.chunk_count,
            }
            for rs in repos
        ],
    }

    if as_json:
        _json_output(data)
    else:
        click.echo(f"Repos:   {data['total_repos']}")
        click.echo(f"Vectors: {data['total_vectors']}")
        if repos:
            click.echo()
            click.echo(f"{'REPO ID':<14} {'STATUS':<10} {'CHUNKS':>7}  {'URL'}")
            click.echo("-" * 70)
            for r in data["repos"]:
                click.echo(f"{r['repo_id']:<14} {r['status']:<10} {r['chunk_count']:>7}  {r['url']}")


@cli.command()
@click.argument("query")
@click.option("-k", "--top-k", default=10, show_default=True, type=int, help="Number of results.")
@click.option("-r", "--repo-ids", multiple=True, help="Filter to specific repos (repeatable).")
@click.option("-t", "--file-types", multiple=True, help="Filter by file extension, e.g. .py (repeatable).")
@click.option("--rerank", is_flag=True, help="Apply cross-encoder reranking.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@pass_app
def query(app: AppContext, query: str, top_k: int, repo_ids: tuple[str, ...], file_types: tuple[str, ...], rerank: bool, as_json: bool) -> None:
    """Semantic search across knowledge bases."""
    repo_ids_list = list(repo_ids) if repo_ids else None
    file_types_list = list(file_types) if file_types else None

    results = _run(app.retriever.retrieve(
        query=query,
        top_k=top_k * 3 if rerank else top_k,
        repo_ids=repo_ids_list,
        file_types=file_types_list,
    ))

    if rerank and results:
        results = _run(app.reranker.rerank(query, results, top_n=top_k))

    if not results:
        if as_json:
            _json_output([])
        else:
            click.echo("No results found. Make sure you have repos registered and synced.")
        return

    if as_json:
        _json_output([r.model_dump() for r in results])
        return

    click.echo(f'Query: "{query}"')
    click.echo(f"Found {len(results)} relevant chunks:\n")

    for i, r in enumerate(results, 1):
        repo = r.metadata.get("repo_id", "?")
        source = r.metadata.get("source_file", "?")
        lang = r.metadata.get("language", "")
        score = f"{r.score:.4f}"
        chunk_idx = r.metadata.get("chunk_index", "?")

        click.echo(f"  {i}. {source}  (repo: {repo}, score: {score}, chunk: {chunk_idx}, lang: {lang})")

        text_preview = r.text[:300].replace("\n", "\n     ")
        if len(r.text) > 300:
            text_preview += "..."
        click.echo(f"     {text_preview}")
        click.echo()


@cli.command()
@click.option("--chunk-size", type=int, help="Chunk size in tokens (64–8192).")
@click.option("--chunk-overlap", type=int, help="Overlap in tokens between chunks.")
@click.option("--embedder-provider", type=click.Choice(["sentence_transformers", "openai"]), help="Embedder provider.")
@click.option("--embedder-model", type=str, help="Embedding model name.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@pass_app
def config(app: AppContext, chunk_size: Optional[int], chunk_overlap: Optional[int], embedder_provider: Optional[str], embedder_model: Optional[str], as_json: bool) -> None:
    """View or update pipeline settings."""
    changes = []

    if chunk_size is not None:
        app.config.chunk.chunk_size = chunk_size
        app.chunker = Chunker(app.config.chunk)
        changes.append(f"chunk_size -> {chunk_size}")

    if chunk_overlap is not None:
        app.config.chunk.overlap = chunk_overlap
        app.chunker = Chunker(app.config.chunk)
        changes.append(f"chunk_overlap -> {chunk_overlap}")

    if embedder_provider is not None:
        changes.append(f"embedder_provider -> {embedder_provider} (requires restart)")

    if embedder_model is not None:
        changes.append(f"embedder_model -> {embedder_model} (requires restart)")

    current = {
        "chunk_size": app.config.chunk.chunk_size,
        "chunk_overlap": app.config.chunk.overlap,
        "embedder_provider": app.config.embedder.provider.value,
        "embedder_model": app.config.embedder.model_name,
        "store_backend": app.config.store.backend.value,
        "repos_dir": app.config.repos_dir,
        "persist_dir": app.config.store.persist_dir,
    }

    if as_json:
        data = {"current_config": current}
        if changes:
            data["changes_applied"] = changes
        _json_output(data)
        return

    if changes:
        click.echo("Changes applied:")
        for c in changes:
            click.echo(f"  {c}")
        click.echo()

    click.echo("Current configuration:")
    for key, val in current.items():
        click.echo(f"  {key}: {val}")


@cli.command()
@click.argument("repo_id", required=False, default=None)
@click.option("--force", is_flag=True, help="Force re-bootstrap even if already done.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@pass_app
def bootstrap(app: AppContext, repo_id: Optional[str], force: bool, as_json: bool) -> None:
    """Run LLM agent swarm to generate documentation for a repo."""
    from bootstrap.pipeline import run_bootstrap

    if repo_id:
        repo_state = app.state.get_repo(repo_id)
        if not repo_state:
            raise click.ClickException(f"Repository '{repo_id}' not found.")
        repos = [repo_state]
    else:
        repos = app.state.list_repos()

    if not repos:
        raise click.ClickException("No repositories registered. Use 'add' first.")

    callback = None if as_json else _progress_echo
    results = []

    for rs in repos:
        if not as_json:
            click.echo(f"Bootstrapping {rs.config.repo_id} ({rs.config.url})...")

        result = _run(run_bootstrap(
            repo_config=rs.config,
            pipeline_config=app.config,
            embedder=app.embedder,
            store=app.store,
            chunker=app.chunker,
            state_mgr=app.state,
            force=force,
            progress_callback=callback,
        ))
        results.append(result)

    if as_json:
        _json_output(results if len(results) > 1 else results[0])
    else:
        for r in results:
            status = r.get("status", "?")
            if status == "complete":
                click.echo(f"  {r['repo_id']}: {r.get('entities', 0)} entities, "
                           f"{r.get('docs_generated', 0)} docs, "
                           f"{r.get('chunks_created', 0)} chunks embedded "
                           f"({r.get('tokens_consumed', 0)} tokens used)")
            elif status == "already_bootstrapped":
                click.echo(f"  {r['repo_id']}: already bootstrapped. Use --force to redo.")
            elif status == "error":
                click.echo(f"  {r['repo_id']}: ERROR — {r.get('error', 'unknown')}", err=True)


@cli.command("map")
@click.argument("repo_id")
@click.option("--entities-only", is_flag=True, help="Show only entities, no connections.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@pass_app
def show_map(app: AppContext, repo_id: str, entities_only: bool, as_json: bool) -> None:
    """View the bootstrap knowledge map for a repo."""
    from bootstrap.pipeline import load_bootstrap_map

    data_dir = str(Path(app.config.store.persist_dir).parent)
    bmap = load_bootstrap_map(data_dir, repo_id)

    if not bmap:
        raise click.ClickException(
            f"No bootstrap map found for '{repo_id}'. Run 'bootstrap {repo_id}' first."
        )

    if as_json:
        _json_output(bmap.model_dump())
        return

    click.echo(f"Bootstrap Map: {bmap.repo_id}")
    click.echo(f"  URL:        {bmap.repo_url}")
    click.echo(f"  Status:     {bmap.status.value}")
    click.echo(f"  Created:    {bmap.created_at}")
    click.echo(f"  Entities:   {len(bmap.entities)}")
    click.echo(f"  Connections: {len(bmap.connections)}")
    click.echo(f"  Unknowns:   {len(bmap.unresolved_unknowns)}")
    click.echo(f"  Tokens:     {bmap.total_tokens_consumed}")
    click.echo()

    from collections import defaultdict
    by_kind: dict[str, list] = defaultdict(list)
    for ent in bmap.entities:
        by_kind[ent.kind.value].append(ent)

    for kind, entities in sorted(by_kind.items()):
        click.echo(f"  [{kind.upper()}] ({len(entities)})")
        for e in entities:
            sig = f" — {e.signature}" if e.signature else ""
            click.echo(f"    {e.name}{sig}")
            if e.description:
                click.echo(f"      {e.description}")
            click.echo(f"      {e.file_path}")
        click.echo()

    if not entities_only and bmap.connections:
        click.echo("  [CONNECTIONS]")
        id_to_name: dict[str, str] = {e.entity_id: e.name for e in bmap.entities}
        for conn in bmap.connections:
            src = id_to_name.get(conn.source_entity_id, conn.source_entity_id)
            tgt = id_to_name.get(conn.target_entity_id, conn.target_entity_id)
            click.echo(f"    {src} --[{conn.kind.value}]--> {tgt}")
            if conn.description:
                click.echo(f"      {conn.description}")
        click.echo()

    if bmap.unresolved_unknowns:
        click.echo(f"  [UNRESOLVED UNKNOWNS] ({len(bmap.unresolved_unknowns)})")
        for unk in bmap.unresolved_unknowns:
            click.echo(f"    [{unk.priority}] {unk.description}")
            if unk.file_path:
                click.echo(f"       {unk.file_path}")


if __name__ == "__main__":
    cli()
