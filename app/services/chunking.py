import re
from typing import List, Optional

STRATEGIES = {"fixed", "paragraph"}

# Split after sentence-ending punctuation (optionally followed by closing quotes/brackets).
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
# A line that ends a sentence/heading/list item; the next line then starts a new paragraph.
PARAGRAPH_END = re.compile(r"[.!?:;\"')\]]$")
BLANK_LINE = re.compile(r"\n\s*\n")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [sentence for sentence in SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs.

    Blank lines always separate paragraphs. PDF text from PyMuPDF usually has only
    single newlines (one per visual line), so inside a block a newline is treated
    as a paragraph break only when the line ends a sentence; otherwise the lines
    are a wrapped paragraph and are joined (de-hyphenating split words).
    """
    paragraphs = []

    for block in BLANK_LINE.split(text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        current = ""

        for line in lines:
            if not current:
                current = line
            elif current.endswith("-") and line[:1].islower():
                current = current[:-1] + line
            else:
                current = f"{current} {line}"

            if PARAGRAPH_END.search(current):
                paragraphs.append(current)
                current = ""

        if current:
            paragraphs.append(current)

    return paragraphs


class Chunker:
    """Token-aware chunking using the embedding model's own tokenizer."""

    def __init__(self, tokenizer, max_tokens: int) -> None:
        # max_tokens excludes special tokens ([CLS]/[SEP]), i.e. the usable content length.
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def validate(self, strategy: str, chunk_size: int, overlap: int) -> None:
        if strategy not in STRATEGIES:
            raise ValueError("Invalid chunking strategy. Use 'fixed' or 'paragraph'.")
        if not 1 <= chunk_size <= self.max_tokens:
            raise ValueError(
                f"chunk_size must be between 1 and {self.max_tokens} tokens "
                "(the embedding model's limit)."
            )
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

    def chunk(self, text: str, strategy: str, chunk_size: int, overlap: int = 0) -> List[str]:
        strategy = strategy.lower()
        self.validate(strategy, chunk_size, overlap)

        if strategy == "fixed":
            chunks = self._pack(split_sentences(text), chunk_size, overlap)
        else:
            chunks = []
            for paragraph in self._pack_paragraphs(split_paragraphs(text), chunk_size):
                chunks.extend(paragraph)

        # Safety net: token counts of joined text can differ slightly from the
        # sum of the parts, so hard-split anything that still ends up too long.
        safe_chunks = []
        for chunk in chunks:
            if self.count_tokens(chunk) > chunk_size:
                safe_chunks.extend(self._split_by_tokens(chunk, chunk_size, 0))
            else:
                safe_chunks.append(chunk)

        return [chunk for chunk in safe_chunks if chunk.strip()]

    def _pack_paragraphs(self, paragraphs: List[str], chunk_size: int) -> List[List[str]]:
        """Merge small paragraphs up to chunk_size; split oversized ones by sentence."""
        groups: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if current:
                groups.append(["\n\n".join(current)])
            current, current_tokens = [], 0

        for paragraph in paragraphs:
            tokens = self.count_tokens(paragraph)

            if tokens > chunk_size:
                flush()
                groups.append(self._pack(split_sentences(paragraph), chunk_size, 0))
                continue

            if current and current_tokens + tokens > chunk_size:
                flush()

            current.append(paragraph)
            current_tokens += tokens

        flush()
        return groups

    def _pack(self, sentences: List[str], chunk_size: int, overlap: int) -> List[str]:
        """Greedily pack sentences into chunks of at most chunk_size tokens.

        With overlap > 0, each new chunk starts with the trailing sentences of the
        previous chunk that fit within `overlap` tokens.
        """
        units: List[tuple] = []
        for sentence in sentences:
            tokens = self.count_tokens(sentence)
            if tokens > chunk_size:
                for piece in self._split_by_tokens(sentence, chunk_size, overlap):
                    units.append((piece, self.count_tokens(piece)))
            else:
                units.append((sentence, tokens))

        chunks: List[str] = []
        current: List[tuple] = []
        current_tokens = 0

        for unit in units:
            if current and current_tokens + unit[1] > chunk_size:
                chunks.append(" ".join(text for text, _ in current))

                carried: List[tuple] = []
                carried_tokens = 0
                for previous in reversed(current):
                    if carried_tokens + previous[1] > overlap:
                        break
                    carried.insert(0, previous)
                    carried_tokens += previous[1]

                # Never let the carried overlap push the next chunk over the limit.
                while carried and carried_tokens + unit[1] > chunk_size:
                    carried_tokens -= carried.pop(0)[1]

                current, current_tokens = carried, carried_tokens

            current.append(unit)
            current_tokens += unit[1]

        if current:
            chunks.append(" ".join(text for text, _ in current))

        return chunks

    def _split_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Hard-split text into windows of chunk_size tokens, slicing the original
        string via offsets so casing and punctuation are preserved."""
        offsets = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]

        step = chunk_size - overlap
        pieces = []
        for start in range(0, len(offsets), step):
            window = offsets[start:start + chunk_size]
            pieces.append(text[window[0][0]:window[-1][1]].strip())
            if start + chunk_size >= len(offsets):
                break

        return [piece for piece in pieces if piece]


def resolve_chunk_params(
    strategy: str,
    chunk_size: Optional[int],
    overlap: Optional[int],
    default_chunk_size: int,
    default_overlap: int,
) -> tuple:
    """Fill in defaults. Overlap only applies to the fixed strategy."""
    size = chunk_size if chunk_size is not None else default_chunk_size
    if strategy == "paragraph":
        return size, 0
    return size, overlap if overlap is not None else min(default_overlap, max(size - 1, 0))
