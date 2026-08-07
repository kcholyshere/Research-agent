"""PDF parsing via Docling, trimmed from Finrag's version (ADR-0001).

Finrag's parse.py also generated per-page picture images (`generate_picture_
images`/`images_scale`) for its ColPali vision pipeline and fed table/image
records through an LLM-enrichment step (src/ingestion/enrich.py) for
retrieval-friendly summaries. Phase 1 needs neither - table structure
detection (`do_table_structure`, giving `table.export_to_markdown()`) is a
Docling default, not a ColPali-only add-on, so it stays; picture extraction
and LLM summarisation are dropped along with the image pipeline entirely.
"""

import hashlib
import json
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument, TableItem

from src import config

# Hoisted to a module-level constant (rather than built inline inside
# parse_pdf as before) so _cache_meta reads the exact same options object
# that the converter is given - the two cannot drift apart the way a
# separately-constructed PdfPipelineOptions() in each place could.
PDF_PIPELINE_OPTIONS = PdfPipelineOptions()


def _cache_paths(pdf_path: Path) -> tuple[Path, Path]:
    json_path = config.INTERIM_DIR / f"{pdf_path.stem}.docling.json"
    return json_path, json_path.with_suffix(".meta.json")


def _cache_meta(pdf_path: Path) -> dict:
    """Cache-invalidation key for the parsed-document JSON (finding 15b).

    Previously keyed on the PDF's sha1 alone, so flipping do_table_structure
    or do_ocr on PDF_PIPELINE_OPTIONS above for a local experiment - without
    touching the PDF - would silently serve back a .docling.json parsed
    under the *old* settings. Table structure detection in particular
    changes what extract_table_records sees (table.export_to_markdown
    depends on it being on), so a stale cache there is not just a dev
    inconvenience, it's a wrong parse being reused unnoticed.

    Deliberately naming do_table_structure and do_ocr explicitly rather than
    hashing PdfPipelineOptions().model_dump() wholesale: a full dump also
    captures dozens of unrelated nested settings (OCR engine internals,
    picture-classification model revisions, accelerator thread counts) that
    can change between docling versions with no change to this file, which
    would spuriously invalidate the cache - and re-parsing this ~150-page
    PDF is slow enough that that would be its own hazard. The trade-off:
    this key is stable across docling upgrades but only tracks the two
    fields actually named as mutable here; if PDF_PIPELINE_OPTIONS above
    ever gains a third field someone actually toggles, it needs adding here
    too - it will not be caught automatically.
    """
    return {
        "pdf_sha1": hashlib.sha1(pdf_path.read_bytes()).hexdigest(),
        "do_table_structure": PDF_PIPELINE_OPTIONS.do_table_structure,
        "do_ocr": PDF_PIPELINE_OPTIONS.do_ocr,
    }


def parse_pdf(pdf_path: Path) -> DoclingDocument:
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=PDF_PIPELINE_OPTIONS)}
    )
    document = converter.convert(pdf_path).document

    json_path, meta_path = _cache_paths(pdf_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    document.save_as_json(json_path)
    meta_path.write_text(json.dumps(_cache_meta(pdf_path), indent=2))
    return document


def load_or_parse_pdf(pdf_path: Path) -> DoclingDocument:
    """Re-parse only if the cache is missing or the source PDF has changed
    since - Docling parsing a ~150-page report is slow enough to make caching
    worthwhile even for a single-document corpus."""
    json_path, meta_path = _cache_paths(pdf_path)
    current = _cache_meta(pdf_path)

    if json_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == current:
        return DoclingDocument.load_from_json(json_path)

    return parse_pdf(pdf_path)


def _build_item_to_section(document: DoclingDocument) -> dict[str, str | None]:
    """Map each table's self_ref to the section header preceding it, walking
    the document once in true reading order (see Finrag's ADR log for why a
    page-keyed map is wrong whenever a page holds more than one section)."""
    item_to_section: dict[str, str | None] = {}
    current_section = None

    for item, _level in document.iterate_items():
        if str(getattr(item, "label", "")) == "section_header":
            current_section = item.text
        elif isinstance(item, TableItem):
            item_to_section[item.self_ref] = current_section

    return item_to_section


def extract_table_records(document: DoclingDocument) -> list[dict]:
    """Flatten the parsed document's tables into markdown records with page/section metadata."""
    item_to_section = _build_item_to_section(document)
    records = []

    for table in document.tables:
        page_no = table.prov[0].page_no if table.prov else None
        records.append(
            {
                "text": table.export_to_markdown(document),
                "caption": table.caption_text(document) or None,
                "page": page_no,
                "section": item_to_section.get(table.self_ref),
            }
        )

    return records


def extract_text_records(document: DoclingDocument) -> list[dict]:
    """Flatten the parsed document into text records with page/section metadata."""
    records = []
    current_section = None

    for item in document.texts:
        if item.label == "section_header":
            current_section = item.text

        page_no = item.prov[0].page_no if item.prov else None
        records.append(
            {
                "text": item.text,
                "label": str(item.label),
                "page": page_no,
                "section": current_section,
            }
        )

    return records
