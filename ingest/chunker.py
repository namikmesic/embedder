from __future__ import annotations

import logging
import uuid
from typing import Any

import tiktoken

from config.models import ChunkConfig
from domain.document import Chunk, Document

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, config: ChunkConfig):
        self.chunk_size = config.chunk_size
        self.overlap = config.overlap
        self._enc = tiktoken.get_encoding("cl100k_base")

    def chunk_document(self, doc: Document) -> list[Chunk]:
        tokens = self._enc.encode(doc.text)
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=uuid.uuid4().hex[:16],
                    text=doc.text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "token_count": total_tokens,
                    },
                )
            ]

        chunks: list[Chunk] = []
        step = max(self.chunk_size - self.overlap, 1)
        start = 0
        idx = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self._enc.decode(chunk_tokens)

            chunks.append(
                Chunk(
                    chunk_id=uuid.uuid4().hex[:16],
                    text=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "token_start": start,
                        "token_end": end,
                        "token_count": len(chunk_tokens),
                    },
                )
            )

            start += step
            idx += 1

        for c in chunks:
            c.metadata["total_chunks"] = len(chunks)

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Chunked %d documents into %d chunks", len(docs), len(all_chunks))
        return all_chunks
