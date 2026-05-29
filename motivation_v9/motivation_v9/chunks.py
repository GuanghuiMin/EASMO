"""Natural-language chunk segmentation + ablation (spec §8, §9).

Segmentation rule:
  1. Split on bullet/numbered lines (`^\s*[-*•·]|^\s*\d+[\.)]`).
  2. If no bullets, split into sentences (greedy on .?!).
  3. Merge chunks shorter than 20 chars into the previous chunk.
  4. Cap chunks per context at ``max_chunks`` by merging the shortest
     adjacent pairs.

Chunk text remains exactly equal to the substring of the input
context that produced it (modulo whitespace), so removing a chunk
just drops its substring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


_BULLET_RE = re.compile(r"^\s*[\-\*\u2022\u00b7]|^\s*\d+[\.)]\s+", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    candidate_id: str
    case_id: str
    chunk_index: int
    chunk_text: str
    chunk_chars: int
    chunk_tokens_est: int
    char_span_start: int
    char_span_end: int


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])[ \t]+(?=[A-Z\(\[])", text)
    return [p.strip() for p in parts if p.strip()]


def _split_bullets(text: str) -> List[tuple]:
    """Return list of (start, end, chunk_text) by splitting on bullet lines.

    Each bullet/numbered line begins a new chunk; non-bullet preceding text
    becomes its own chunk.
    """
    lines = text.split("\n")
    chunks: List[tuple] = []
    current: List[str] = []
    cursor = 0
    chunk_start = 0
    for ln in lines:
        ln_with_nl = ln + "\n"
        if _BULLET_RE.match(ln_with_nl) or _BULLET_RE.match(ln):
            if current:
                joined = "\n".join(current).strip()
                if joined:
                    chunks.append((chunk_start, cursor, joined))
                current = []
                chunk_start = cursor
            current = [ln]
            cursor += len(ln_with_nl)
        else:
            current.append(ln)
            cursor += len(ln_with_nl)
    if current:
        joined = "\n".join(current).strip()
        if joined:
            chunks.append((chunk_start, len(text), joined))
    return chunks


def segment_chunks(
    *,
    candidate_id: str,
    case_id: str,
    text: str,
    min_chunk_chars: int = 20,
    max_chunks: int = 12,
) -> List[Chunk]:
    """Segment ``text`` into a list of ``Chunk``s.

    Uses bullets if present; else sentences. Merges short chunks and
    caps the total. We do NOT preserve exact char spans across the
    merging step (the chunk text is the authoritative source) — the
    ablation only needs to remove the chunk substring from the input.
    """
    # Stage 1: bullets vs sentences
    if _BULLET_RE.search(text):
        bullet_groups = _split_bullets(text)
        raw_chunks = [g[2] for g in bullet_groups]
    else:
        raw_chunks = _split_sentences(text)

    # Stage 2: merge short chunks into the previous chunk
    merged: List[str] = []
    for c in raw_chunks:
        if merged and len(c) < min_chunk_chars:
            merged[-1] = merged[-1].rstrip() + " " + c.strip()
        else:
            merged.append(c.strip())
    # second pass: if the first chunk is short, merge into next
    if merged and len(merged[0]) < min_chunk_chars and len(merged) >= 2:
        merged[1] = merged[0].strip() + " " + merged[1].strip()
        merged = merged[1:]

    # Stage 3: cap to ``max_chunks`` by merging the shortest adjacent pair
    while len(merged) > max_chunks:
        best_i = 0
        best_sum = len(merged[0]) + len(merged[1])
        for i in range(1, len(merged) - 1):
            s = len(merged[i]) + len(merged[i + 1])
            if s < best_sum:
                best_sum = s
                best_i = i
        merged[best_i] = (merged[best_i].rstrip() + " " + merged[best_i + 1].lstrip()).strip()
        merged.pop(best_i + 1)

    # Stage 4: emit Chunk dataclasses with rough char spans
    out: List[Chunk] = []
    cursor = 0
    for i, c in enumerate(merged):
        # find c in text starting at cursor (best-effort)
        idx = text.find(c[: min(40, len(c))], cursor) if c else -1
        if idx >= 0:
            start = idx
            end = idx + len(c)
            cursor = end
        else:
            start = cursor
            end = cursor + len(c)
            cursor = end
        out.append(Chunk(
            chunk_id=f"{candidate_id}__chunk_{i:02d}",
            candidate_id=candidate_id,
            case_id=case_id,
            chunk_index=i,
            chunk_text=c,
            chunk_chars=len(c),
            chunk_tokens_est=max(1, len(c) // 4),
            char_span_start=start,
            char_span_end=end,
        ))
    return out


def remove_chunk(text: str, chunk: Chunk) -> str:
    """Return ``text`` with ``chunk.chunk_text`` removed (first occurrence)."""
    needle = chunk.chunk_text
    if not needle:
        return text
    idx = text.find(needle)
    if idx < 0:
        # fallback: try the first 30 chars
        idx = text.find(needle[: min(30, len(needle))])
    if idx < 0:
        return text
    end = idx + len(needle)
    before = text[:idx].rstrip(" \n")
    after = text[end:].lstrip(" \n")
    if before and after:
        return before + "\n" + after
    return before + after


__all__ = [
    "Chunk",
    "segment_chunks",
    "remove_chunk",
]
