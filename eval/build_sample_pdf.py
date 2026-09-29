"""Render eval/sources/helio_h5_manual.txt to eval/docs/helio_h5_manual.pdf.

The PDF is committed, so this only needs to be re-run after editing the source.
Lines starting with "# " become headings; blank lines separate paragraphs.

    python -m eval.build_sample_pdf
"""

import html
from pathlib import Path

import fitz  # PyMuPDF

EVAL_DIR = Path(__file__).parent
SOURCE = EVAL_DIR / "sources" / "helio_h5_manual.txt"
OUTPUT = EVAL_DIR / "docs" / "helio_h5_manual.pdf"

CSS = """
body { font-family: sans-serif; font-size: 10.5pt; }
h1 { font-size: 16pt; margin-bottom: 6pt; }
h2 { font-size: 12.5pt; margin-top: 12pt; margin-bottom: 4pt; }
p { margin-bottom: 6pt; }
"""


def source_to_html(text: str) -> str:
    parts = []
    for index, block in enumerate(b.strip() for b in text.split("\n\n")):
        if not block:
            continue
        if block.startswith("# "):
            tag = "h1" if index == 0 else "h2"
            parts.append(f"<{tag}>{html.escape(block[2:])}</{tag}>")
        else:
            parts.append(f"<p>{html.escape(' '.join(block.split()))}</p>")
    return "\n".join(parts)


def build() -> None:
    story = fitz.Story(html=source_to_html(SOURCE.read_text(encoding="utf-8")), user_css=CSS)
    page_rect = fitz.paper_rect("a4")
    content_rect = page_rect + (60, 60, -60, -60)

    writer = fitz.DocumentWriter(str(OUTPUT))
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
