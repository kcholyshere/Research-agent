"""Regression tests for audit.md findings 10, 15b and 15c.

All three are ingestion/embedding-time defects that are silent by
construction - a wrong or missing signal, not a raised exception - which is
exactly why each needs a test that pins the *presence* of the right output,
not merely the absence of a crash. See each finding's section in
agent_docs/audit.md for the full failure scenario; the summaries below are
just enough to explain what each test is asserting.

Finding 10: `GeminiEmbeddings.embed_documents` read only `e.values` from the
embedding API response, discarding `e.statistics.truncated` - the only
signal that an oversized table chunk (chunk.py never splits a table) was cut
down before being embedded. The stored page_content is unaffected, so
nothing looks wrong until a query into the chunk's truncated tail fails to
retrieve it. Fixed by embedder.py's `_report_truncation`, which distinguishes
three states (truncated / not truncated / unknown - `statistics` itself is
documented "Gemini Enterprise Agent Platform only" and may come back None on
this Vertex deployment) rather than collapsing "unknown" into "fine".

Finding 15b: `parse.py`'s Docling cache keyed on the PDF's sha1 alone, so
toggling `do_table_structure` or `do_ocr` on `PDF_PIPELINE_OPTIONS` without
touching the PDF served a stale `.docling.json` parsed under the old
settings. Fixed by folding both fields into `_cache_meta`.

Finding 15c: `chunk.py`'s `_group_into_sections` guarded `end_page` against a
`None` page overwriting an already-known value, but seeded `start_page`
unconditionally from the first record - a section whose opening record(s)
lack Docling provenance kept `start_page = None` forever even once a later
record in the same section supplied a real page, which a citation renderer
could show as "page None-42". Fixed by backfilling `start_page` the same way.

Everything here is offline: no Vertex call, no Docling parse, no real PDF.
`tmp_path` stands in for anything that would otherwise touch data/.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.embedding import embedder
from src.embedding.embedder import GeminiEmbeddings
from src.ingestion import parse
from src.ingestion.chunk import _group_into_sections


# --- Finding 10: embedding truncation is surfaced, not discarded --------


class _FakeStatistics:
    def __init__(self, truncated: bool | None) -> None:
        self.truncated = truncated


class _FakeEmbedding:
    def __init__(self, values: list[float], statistics: _FakeStatistics | None) -> None:
        self.values = values
        self.statistics = statistics


class _FakeResponse:
    def __init__(self, embeddings: list[_FakeEmbedding]) -> None:
        self.embeddings = embeddings


class _FakeModels:
    """Stands in for `client.models`, returning one canned response per call
    regardless of the batch it is given - enough to reach the per-embedding
    statistics handling in embed_documents without a real Vertex client."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def embed_content(self, *, model, contents, config):  # noqa: ANN001 - matches genai signature
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.models = _FakeModels(response)


def test_truncated_chunk_produces_a_visible_report(capsys: pytest.CaptureFixture) -> None:
    """A chunk whose statistics say truncated=True must be named in the
    build-time report, not silently dropped like the old code dropped it."""
    response = _FakeResponse(
        [
            _FakeEmbedding([0.1, 0.2], _FakeStatistics(truncated=False)),
            _FakeEmbedding([0.3, 0.4], _FakeStatistics(truncated=True)),
        ]
    )
    with patch.object(embedder, "get_client", return_value=_FakeClient(response)):
        vectors = GeminiEmbeddings().embed_documents(["short chunk", "huge table chunk"])

    # The old, wrong behaviour ("return values, discard statistics") still
    # has to keep working - this is detection layered on top, not a remedy.
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]

    stderr = capsys.readouterr().err
    assert "TRUNCATION" in stderr
    assert "1/2" in stderr  # one of the two chunks confirmed truncated
    assert "[1]" in stderr  # the truncated chunk's index, for naming which one


def test_missing_statistics_is_reported_as_unknown_not_success(
    capsys: pytest.CaptureFixture,
) -> None:
    """`statistics` is documented Gemini-Enterprise-only in the installed
    google-genai package, so on this Vertex deployment it may come back None
    for every chunk. That must read as "couldn't check", never as "checked
    and fine" - collapsing the two would be a false all-clear on exactly the
    deployment where this project actually runs.
    """
    response = _FakeResponse(
        [
            _FakeEmbedding([0.1, 0.2], None),
            _FakeEmbedding([0.3, 0.4], None),
        ]
    )
    with patch.object(embedder, "get_client", return_value=_FakeClient(response)):
        GeminiEmbeddings().embed_documents(["chunk a", "chunk b"])

    stderr = capsys.readouterr().err
    assert "UNKNOWN" in stderr
    assert "2/2" in stderr


# --- Finding 15b: pipeline options are part of the parse cache key ------


def test_different_pipeline_options_produce_different_cache_keys(tmp_path: Path) -> None:
    """Two PDFs with identical bytes but different PDF_PIPELINE_OPTIONS must
    key the parse cache differently - otherwise toggling do_table_structure
    or do_ocr without touching the PDF serves back a stale parse from before
    the toggle, which is exactly finding 15b's failure scenario."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 not a real pdf, only bytes matter here")

    class _Options:
        def __init__(self, do_table_structure: bool, do_ocr: bool) -> None:
            self.do_table_structure = do_table_structure
            self.do_ocr = do_ocr

    with patch.object(parse, "PDF_PIPELINE_OPTIONS", _Options(do_table_structure=True, do_ocr=True)):
        meta_a = parse._cache_meta(pdf_path)
    with patch.object(parse, "PDF_PIPELINE_OPTIONS", _Options(do_table_structure=False, do_ocr=True)):
        meta_b = parse._cache_meta(pdf_path)

    assert meta_a != meta_b
    # Same PDF bytes, so the sha1 half of the key is identical - the
    # difference must come entirely from the pipeline options, confirming
    # they are actually part of the key rather than a coincidental miss.
    assert meta_a["pdf_sha1"] == meta_b["pdf_sha1"]
    assert meta_a["do_table_structure"] != meta_b["do_table_structure"]

    # And the key must round-trip through JSON the same way the real cache
    # file does (json.dumps at write time, json.loads at read time).
    assert json.loads(json.dumps(meta_a)) == meta_a


# --- Finding 15c: a missing start_page cannot leak "None" into a citation


def _make_record(section: str, page: int | None, text: str = "x") -> dict:
    return {"label": "text", "section": section, "page": page, "text": text}


def test_section_with_no_provenance_on_its_first_record_backfills_start_page() -> None:
    """The first record of a section has no Docling provenance (page=None);
    a later record in the SAME section does. Before the fix, start_page was
    seeded None from the first record and never updated, so a citation
    built from this section would render "page None-42" even though a real
    page number was available in the same section."""
    records = [
        _make_record("Introduction", page=None),
        _make_record("Introduction", page=42),
    ]

    sections = _group_into_sections(records)

    assert len(sections) == 1
    assert sections[0]["start_page"] == 42
    assert sections[0]["end_page"] == 42

    citation = f"page {sections[0]['start_page']}-{sections[0]['end_page']}"
    assert "None" not in citation


def test_section_with_no_provenance_anywhere_stays_symmetric() -> None:
    """If no record in the section ever carries a page, start_page and
    end_page must both stay None (an honest "page unknown") rather than one
    of them silently picking up a real number while the other stays None."""
    records = [
        _make_record("Appendix", page=None),
        _make_record("Appendix", page=None),
    ]

    sections = _group_into_sections(records)

    assert sections[0]["start_page"] is None
    assert sections[0]["end_page"] is None


# --- Oversized tables are split by row, keeping every part self-describing ---

_TABLE_HEADER = "| Member | Amount paid |"
_TABLE_SEPARATOR = "|--------|-------------|"


def _wide_table(rows: int, row_width: int = 200) -> str:
    body = [f"| {'Country' + str(i):<{row_width}} | {i * 111:<20} |" for i in range(rows)]
    return "\n".join([_TABLE_HEADER, _TABLE_SEPARATOR, *body])


def test_a_table_that_fits_is_not_split() -> None:
    """The original design decision still holds wherever it can."""
    from src.ingestion.chunk import _split_table_text

    text = _wide_table(3)
    assert _split_table_text(text, 6000) == [text]


def test_every_part_of_a_split_table_repeats_the_header() -> None:
    """This is what makes splitting better than truncating rather than worse.

    A part carrying bare `| value | value |` rows with no column names is less
    retrievable than the truncated original, not more - so the header block is
    the whole point of splitting by row instead of by character.
    """
    from src.ingestion.chunk import _split_table_text

    parts = _split_table_text(_wide_table(120), 6000)

    assert len(parts) > 1, "a 120-row wide table should not fit in 6000 characters"
    for part in parts:
        assert part.startswith(_TABLE_HEADER), "a part without column names is barely retrievable"
        assert _TABLE_SEPARATOR in part


def test_splitting_loses_no_rows_and_duplicates_none() -> None:
    from src.ingestion.chunk import _split_table_text

    original = _wide_table(120)
    parts = _split_table_text(original, 6000)

    body = [line for line in original.split("\n")[2:]]
    recovered = [line for part in parts for line in part.split("\n")[2:]]
    assert recovered == body, "row order and count must survive the split exactly"


def test_no_row_is_ever_cut_in_half() -> None:
    """A malformed row would be worse than an oversized part, so the split never cuts one."""
    from src.ingestion.chunk import _split_table_text

    for part in _split_table_text(_wide_table(120), 6000):
        for line in part.split("\n"):
            assert line.startswith("|") and line.endswith("|"), f"cut mid-row: {line!r}"


def test_text_with_no_separator_row_still_degrades_rather_than_raising() -> None:
    """Docling occasionally emits a block that is not a markdown table at all."""
    from src.ingestion.chunk import _split_table_text

    parts = _split_table_text("just a long run of prose " * 500, 6000)
    assert len(parts) > 1
    assert "".join(parts) == "just a long run of prose " * 500
