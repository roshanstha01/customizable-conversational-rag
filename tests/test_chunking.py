import pytest

from app.services.chunking import (
    Chunker,
    normalize_text,
    resolve_chunk_params,
    split_paragraphs,
    split_sentences,
)

LONG_PARAGRAPH = " ".join(
    f"Sentence {i} describes vector search with Qdrant and chat memory with Redis." for i in range(40)
)


def test_normalize_text_rejoins_hyphenated_line_breaks():
    text = "It com-\nbines data and optimi- \n zation. Self-\nService stays split."
    assert normalize_text(text) == "It combines data and optimization. Self-\nService stays split."


def test_split_sentences():
    assert split_sentences("First one. Second one!  Third?\nFourth") == [
        "First one.", "Second one!", "Third?", "Fourth",
    ]


def test_split_paragraphs_handles_pdf_single_newlines():
    pdf_text = (
        "Machine learning lets computers learn\n"
        "from data. It is widely used.\n"
        "Deep learning uses neural networks with many\n"
        "layers.\n"
    )
    assert split_paragraphs(pdf_text) == [
        "Machine learning lets computers learn from data. It is widely used.",
        "Deep learning uses neural networks with many layers.",
    ]


def test_split_paragraphs_uses_blank_lines():
    assert split_paragraphs("First para\nstill first\n\nSecond para") == [
        "First para still first",
        "Second para",
    ]


def test_fixed_chunks_respect_size(chunker):
    chunks = chunker.chunk(LONG_PARAGRAPH, "fixed", chunk_size=40, overlap=0)
    assert len(chunks) > 1
    assert all(chunker.count_tokens(chunk) <= 40 for chunk in chunks)
    # Sentence boundaries are respected: every chunk ends a sentence.
    assert all(chunk.endswith(".") for chunk in chunks)


def test_fixed_chunks_overlap_by_whole_sentences(chunker):
    chunks = chunker.chunk(LONG_PARAGRAPH, "fixed", chunk_size=40, overlap=15)
    first_sentences = split_sentences(chunks[0])
    second_sentences = split_sentences(chunks[1])
    assert first_sentences[-1] == second_sentences[0]
    assert all(chunker.count_tokens(chunk) <= 40 for chunk in chunks)


def test_fixed_uses_normalized_text(chunker):
    chunks = chunker.chunk("Machine learning com-\nbines statistics.", "fixed", chunk_size=50, overlap=0)
    assert chunks == ["Machine learning combines statistics."]


def test_paragraph_merges_small_paragraphs(chunker):
    text = "Short one.\n\nShort two.\n\nShort three."
    assert chunker.chunk(text, "paragraph", chunk_size=50) == ["Short one.\n\nShort two.\n\nShort three."]


def test_paragraph_splits_long_paragraph(chunker):
    text = "Intro paragraph.\n\n" + LONG_PARAGRAPH
    chunks = chunker.chunk(text, "paragraph", chunk_size=40)
    assert chunks[0] == "Intro paragraph."
    assert len(chunks) > 2
    assert all(chunker.count_tokens(chunk) <= 40 for chunk in chunks)


def test_sentence_longer_than_limit_is_hard_split_preserving_case(chunker):
    run_on = " ".join(f"Word{i}" for i in range(100))  # no sentence punctuation
    chunks = chunker.chunk(run_on, "fixed", chunk_size=30, overlap=0)
    assert all(chunker.count_tokens(chunk) <= 30 for chunk in chunks)
    assert chunks[0].startswith("Word0 Word1")
    assert " ".join(chunks) == run_on


@pytest.mark.parametrize("strategy", ["fixed", "paragraph"])
def test_no_chunk_exceeds_model_limit(chunker, strategy):
    text = LONG_PARAGRAPH + "\n\n" + "x " * 1000
    chunks = chunker.chunk(text, strategy, chunk_size=chunker.max_tokens, overlap=0)
    assert all(chunker.count_tokens(chunk) <= chunker.max_tokens for chunk in chunks)


@pytest.mark.parametrize(
    "strategy, chunk_size, overlap, message",
    [
        ("fixed", 300, 0, "chunk_size must be between"),
        ("fixed", 0, 0, "chunk_size must be between"),
        ("fixed", 50, 50, "overlap must be smaller"),
        ("fixed", 50, -1, "overlap cannot be negative"),
        ("semantic", 50, 0, "Invalid chunking strategy"),
    ],
)
def test_validate_rejects_bad_parameters(chunker, strategy, chunk_size, overlap, message):
    with pytest.raises(ValueError, match=message):
        chunker.validate(strategy, chunk_size, overlap)


@pytest.mark.parametrize(
    "strategy, chunk_size, overlap, expected",
    [
        ("fixed", None, None, (200, 40)),
        ("fixed", 100, None, (100, 20)),  # min(40, 100 // 5)
        ("fixed", 3, None, (3, 0)),
        ("fixed", 100, 5, (100, 5)),
        ("paragraph", 100, 30, (100, 0)),  # overlap ignored
    ],
)
def test_resolve_chunk_params(strategy, chunk_size, overlap, expected):
    assert resolve_chunk_params(strategy, chunk_size, overlap, 200, 40) == expected


def test_real_tokenizer_never_exceeds_256_tokens():
    """Uses the real MiniLM tokenizer when it is cached locally; skipped otherwise."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        )
    except Exception:
        pytest.skip("MiniLM tokenizer not available offline")

    real = Chunker(tokenizer, max_tokens=256 - tokenizer.num_special_tokens_to_add())
    text = LONG_PARAGRAPH + "\n\n" + "tokenization " * 600
    for strategy in ("fixed", "paragraph"):
        for chunk in real.chunk(text, strategy, chunk_size=real.max_tokens, overlap=0):
            assert len(tokenizer(chunk)["input_ids"]) <= 256
