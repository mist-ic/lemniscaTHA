"""
ClearPath RAG Chatbot — Structure-Aware Chunker

Implements paragraph-based chunking with:
- Section heading detection (conservative, to avoid splitting tables)
- FAQ Q&A pair preservation (PDFs 17, 21)
- Table handling (keeps rows with headers)
- Configurable chunk size and overlap
- Small-chunk merging to avoid tiny chunks
"""

import re
from dataclasses import dataclass
from typing import List

from app.pipeline.extractor import Document


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    chunk_id: str
    document: str
    page: int
    section_heading: str
    text: str
    token_count: int


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (English)."""
    return max(1, len(text) // 4)


def _is_heading(line: str) -> bool:
    """
    Detect if a line is likely a section heading.
    Conservative: avoids false positives on table cells, single words, etc.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    # Too short to be a heading (avoid table cells like "High", "Low", etc.)
    if len(stripped) < 4:
        return False
    words = stripped.split()
    word_count = len(words)
    # Must be 2-10 words
    if word_count < 2 or word_count > 10:
        return False
    # Skip lines that are just numbers, symbols, or look like table data
    if re.match(r'^[\d\$\%\.\,\-\+]+$', stripped):
        return False
    # Skip lines ending with common sentence punctuation (. , ;)
    if stripped[-1] in '.,:;':
        return False
    # All-caps lines of reasonable length
    if stripped.isupper() and 3 < len(stripped) < 60:
        return True
    # Title-case lines without ending punctuation
    if stripped[0].isupper() and not stripped.endswith(('?', '!')):
        # Check if most words are capitalized (title case)
        cap_words = sum(1 for w in words if w[0].isupper() or w.lower() in {
            'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'by'
        })
        if cap_words >= len(words) * 0.6:
            return True
    return False


def _is_faq_content(text: str) -> bool:
    """Check if text contains FAQ Q&A patterns."""
    q_count = len(re.findall(r'(?m)^Q[:.]', text))
    return q_count >= 2  # at least 2 Q: lines to be considered FAQ


def _extract_faq_pairs(text: str) -> tuple:
    """
    Extract Q&A pairs as atomic chunks.
    Returns (faq_pairs: List[str], remaining_text: str).
    """
    # Split on Q: patterns, keeping the Q: prefix
    parts = re.split(r'(?m)(?=^Q[:.]\s)', text)
    faq_pairs = []
    non_faq_parts = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r'^Q[:.]\s', part):
            faq_pairs.append(part)
        else:
            non_faq_parts.append(part)

    remaining = '\n\n'.join(non_faq_parts)
    return faq_pairs, remaining


def _split_into_sections(text: str) -> List[tuple]:
    """Split text into (heading, body) tuples. Conservative heading detection."""
    lines = text.split('\n')
    sections = []
    current_heading = "General"
    current_lines = []

    for line in lines:
        if _is_heading(line) and len(current_lines) > 0:
            body = '\n'.join(current_lines).strip()
            if body:
                sections.append((current_heading, body))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save final section
    body = '\n'.join(current_lines).strip()
    if body:
        sections.append((current_heading, body))

    return sections if sections else [("General", text)]


def _split_paragraphs(text: str) -> List[str]:
    """Split text by double newlines into paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def _merge_small_paragraphs(paragraphs: List[str], max_tokens: int) -> List[str]:
    """Merge consecutive small paragraphs up to max_tokens."""
    if not paragraphs:
        return []

    merged = []
    buffer = ""

    for para in paragraphs:
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if _estimate_tokens(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            # If a single paragraph exceeds max_tokens, keep it (will be split later)
            buffer = para

    if buffer:
        merged.append(buffer)

    return merged


def _split_long_text(text: str, max_tokens: int) -> List[str]:
    """Split text that exceeds max_tokens by sentences."""
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if _estimate_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks if chunks else [text]


def _is_pricing_or_table_content(text: str) -> bool:
    """Check if text contains pricing/plan data that should be kept in larger chunks."""
    text_lower = text.lower()
    has_dollar = '$' in text
    plan_names = {'free', 'pro', 'enterprise', 'starter', 'basic'}
    has_plan = any(p in text_lower for p in plan_names)
    has_table_markers = text.count('|') >= 3 or text.count('\t') >= 3
    # Price + plan name = pricing table content
    return (has_dollar and has_plan) or has_table_markers


def _apply_overlap(chunks: List[str], overlap_tokens: int) -> List[str]:
    """Apply overlap between consecutive chunks by prepending tail of previous chunk."""
    if len(chunks) <= 1 or overlap_tokens <= 0:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1]
        prev_words = prev_text.split()

        # ~1.3 words per token
        overlap_words = max(1, int(overlap_tokens * 1.3))
        if overlap_words < len(prev_words):
            overlap_text = ' '.join(prev_words[-overlap_words:])
            result.append(overlap_text + " " + chunks[i])
        else:
            result.append(chunks[i])

    return result


def _post_merge_small_chunks(
    chunks: List[tuple],  # list of (chunk_text, section_heading) pairs
    min_tokens: int = 80,
    max_tokens: int = 400,
) -> List[tuple]:
    """
    Final pass: aggressively merge small chunks with neighbors.
    Merges ACROSS section boundaries when chunks are too small.
    Uses a higher max for pricing/table content (500 tokens).
    """
    if not chunks:
        return []

    merged = []
    buffer_text = ""
    buffer_heading = ""

    for text, heading in chunks:
        if not buffer_text:
            buffer_text = text
            buffer_heading = heading
            continue

        buffer_tokens = _estimate_tokens(buffer_text)
        next_tokens = _estimate_tokens(text)
        combined = buffer_text + "\n\n" + text
        combined_tokens = _estimate_tokens(combined)

        # Determine max size — allow larger chunks for pricing/table content
        effective_max = 500 if _is_pricing_or_table_content(combined) else max_tokens

        # Merge if buffer is small OR next chunk is small, and combined fits
        if (buffer_tokens < min_tokens or next_tokens < min_tokens) and combined_tokens <= effective_max:
            buffer_text = combined
            if heading != "General":
                buffer_heading = heading
            continue

        # Buffer is big enough, flush it
        merged.append((buffer_text, buffer_heading))
        buffer_text = text
        buffer_heading = heading

    if buffer_text:
        # If final chunk is still too small, merge with last merged chunk
        if merged and _estimate_tokens(buffer_text) < min_tokens:
            prev_text, prev_heading = merged[-1]
            combined = prev_text + "\n\n" + buffer_text
            effective_max = 500 if _is_pricing_or_table_content(combined) else max_tokens
            if _estimate_tokens(combined) <= effective_max:
                merged[-1] = (combined, prev_heading)
            else:
                merged.append((buffer_text, buffer_heading))
        else:
            merged.append((buffer_text, buffer_heading))

    return merged


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    min_chunk_tokens: int = 80,
) -> List[Chunk]:
    """
    Chunk extracted documents into retrieval-ready pieces.

    Features:
    - Structure-aware splitting by sections and paragraphs
    - FAQ Q&A pairs kept atomic
    - Aggressive small chunk merging (min 80 tokens, across section boundaries)
    - Pricing/table chunks allowed up to 500 tokens
    - Configurable size and overlap

    Args:
        documents: List of Document objects from extractor.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Overlap tokens between adjacent chunks.
        min_chunk_tokens: Minimum tokens per chunk (smaller merged with neighbors).

    Returns:
        List of Chunk objects with metadata.
    """
    all_chunks: List[Chunk] = []

    for doc in documents:
        filename_base = doc.filename.replace('.pdf', '')
        doc_text = doc.text

        # --- FAQ detection ---
        faq_pairs = []
        if _is_faq_content(doc_text):
            faq_pairs, doc_text = _extract_faq_pairs(doc_text)
            for idx, pair in enumerate(faq_pairs):
                chunk_id = f"{filename_base}_p{doc.page_number}_faq{idx}"
                all_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document=doc.filename,
                    page=doc.page_number,
                    section_heading="FAQ",
                    text=pair,
                    token_count=_estimate_tokens(pair),
                ))

        if not doc_text.strip():
            continue

        # --- Section splitting ---
        sections = _split_into_sections(doc_text)

        # Collect all section-level chunks (text, heading) for this document page
        page_chunks = []

        for heading, body in sections:
            # Paragraph-based chunking
            paragraphs = _split_paragraphs(body)
            merged = _merge_small_paragraphs(paragraphs, chunk_size)

            # Split any still-too-long chunks
            final_texts = []
            for m in merged:
                final_texts.extend(_split_long_text(m, chunk_size))

            # Apply overlap within section
            final_texts = _apply_overlap(final_texts, chunk_overlap)

            for text in final_texts:
                page_chunks.append((text, heading))

        # --- Post-merge tiny chunks ---
        page_chunks = _post_merge_small_chunks(page_chunks, min_chunk_tokens, chunk_size)

        # --- Create Chunk objects ---
        for idx, (text, heading) in enumerate(page_chunks):
            chunk_id = f"{filename_base}_p{doc.page_number}_c{idx}"
            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                document=doc.filename,
                page=doc.page_number,
                section_heading=heading,
                text=text,
                token_count=_estimate_tokens(text),
            ))

    return all_chunks


if __name__ == "__main__":
    from app.pipeline.extractor import extract_all_pdfs

    docs = extract_all_pdfs("../docs")
    chunks = chunk_documents(docs)

    print(f"Total documents (pages): {len(docs)}")
    print(f"Total chunks: {len(chunks)}")

    # Token stats
    tokens = [c.token_count for c in chunks]
    print(f"Token range: {min(tokens)} – {max(tokens)}, avg: {sum(tokens)/len(tokens):.0f}")

    # Distribution
    import collections
    ranges = collections.Counter()
    for c in chunks:
        if c.token_count < 10: ranges['<10'] += 1
        elif c.token_count < 50: ranges['10-50'] += 1
        elif c.token_count < 100: ranges['50-100'] += 1
        elif c.token_count < 200: ranges['100-200'] += 1
        elif c.token_count < 300: ranges['200-300'] += 1
        elif c.token_count < 400: ranges['300-400'] += 1
        else: ranges['400+'] += 1

    for k in ['<10', '10-50', '50-100', '100-200', '200-300', '300-400', '400+']:
        print(f"  {k}: {ranges.get(k, 0)}")

    # Show a few chunks
    for c in chunks[:5]:
        print(f"\n--- {c.chunk_id} ({c.token_count} tokens) ---")
        print(f"  Section: {c.section_heading}")
        print(f"  Text: {c.text[:150]}...")
