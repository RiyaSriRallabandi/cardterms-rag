"""PDF text extraction with per-page OCR fallback and table serialisation.

Tables are detected and rendered as Markdown, and the layout blocks they
occupy are excluded from the surrounding body text so that table contents
appear once, in a structured form, rather than as interleaved fragments.

OCR is applied at two levels. Whole pages are recognised when they carry no
text layer. Individual table cells are recognised when a cell extracts empty
on a page containing images: filings mix live text and pasted images within a
single table, and an empty cell would otherwise present a fee label with no
value attached.
"""

import io
from dataclasses import dataclass, field

import fitz

from cardterms.logging import log

# A page yielding fewer characters than this contains no usable text layer.
OCR_CHAR_THRESHOLD = 100

# Render resolution for OCR. Higher improves accuracy on small type at the
# cost of speed; 300 is the usual floor for reliable recognition.
OCR_DPI = 300

# Cell regions smaller than this are rule lines or padding, not content.
MIN_CELL_SIZE_PT = 4

# Candidate regions below these dimensions are usually layout artefacts
# rather than data tables.
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2

# Image regions below this fraction of page area are logos and rules.
MIN_IMAGE_AREA_RATIO = 0.02

# An OCR'd region yielding less than this is noise, not content.
MIN_REGION_OCR_CHARS = 20

# A detected region must have this share of populated cells to be treated as a
# data table. Layout structures such as indented lists and multi-column prose
# are frequently detected as sparse grids, and excluding their blocks from the
# body would remove real prose from the document.
MIN_TABLE_FILL_RATIO = 0.5
MIN_TABLE_FILLED_CELLS = 4

# Common-word density below which a page's text layer is unusable. PDFs with
# damaged font encodings extract as glyph codes: structurally text-like, but
# semantically empty. The rendered page is unaffected, so OCR recovers it.
MIN_WORDLIKE_SCORE = 5.0

# Pages shorter than this carry too few words to judge, and are frequently
# legitimate rate tables with little prose.
WORDLIKE_MIN_CHARS = 300

COMMON_WORDS = (
    " the ",
    " and ",
    " you ",
    " to ",
    " of ",
    " your ",
    " we ",
    " or ",
    " a ",
    " in ",
)


@dataclass
class ParsedTable:
    page_no: int
    index: int
    markdown: str
    n_rows: int
    n_cols: int
    empty_cells: int = 0
    ocr_cells_filled: int = 0


@dataclass
class ParsedPage:
    page_no: int
    text: str
    ocr_applied: bool
    ocr_regions: int = 0
    tables: list[ParsedTable] = field(default_factory=list)


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]

    @property
    def ocr_page_count(self) -> int:
        return sum(page.ocr_applied for page in self.pages)

    @property
    def table_count(self) -> int:
        return sum(len(page.tables) for page in self.pages)


def _ocr_image(image) -> str:
    import pytesseract

    return pytesseract.image_to_string(image, config="--psm 6").strip()


def _render(page: fitz.Page, clip: fitz.Rect | None = None):
    from PIL import Image

    pixmap = page.get_pixmap(dpi=OCR_DPI, clip=clip)
    return Image.open(io.BytesIO(pixmap.tobytes("png")))


def _count_empty(rows: list[list]) -> int:
    return sum(1 for row in rows for cell in row if not (cell or "").strip())


def _fill_empty_cells(page: fitz.Page, table, rows: list[list]) -> None:
    """Recognise cells that extracted empty, using the table's own geometry."""
    try:
        table_rows = table.rows
    except Exception:  # noqa: BLE001 - geometry is not always available
        return

    for row_index, row in enumerate(table_rows):
        if row_index >= len(rows):
            break
        for col_index, bbox in enumerate(getattr(row, "cells", None) or []):
            if bbox is None or col_index >= len(rows[row_index]):
                continue
            if (rows[row_index][col_index] or "").strip():
                continue

            rect = fitz.Rect(bbox)
            if rect.width < MIN_CELL_SIZE_PT or rect.height < MIN_CELL_SIZE_PT:
                continue

            try:
                text = _ocr_image(_render(page, rect))
            except Exception as exc:  # noqa: BLE001 - a failed cell must not stop the page
                log.debug("cell_ocr_failed", page=page.number, error=str(exc))
                continue
            if text:
                rows[row_index][col_index] = text


def _rows_to_markdown(rows: list[list]) -> tuple[str, int, int]:
    """Render extracted table rows as a Markdown table."""
    cleaned = [
        [(cell or "").replace("\n", " ").strip() for cell in row] for row in rows
    ]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return "", 0, 0

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines), len(cleaned), width


def _find_tables(
    page: fitz.Page, page_no: int
) -> tuple[list[ParsedTable], list[fitz.Rect]]:
    try:
        finder = page.find_tables()
    except Exception:  # noqa: BLE001 - table detection is best-effort
        return [], []

    page_has_images = bool(page.get_images(full=True))

    tables: list[ParsedTable] = []
    boxes: list[fitz.Rect] = []
    for index, table in enumerate(finder.tables):
        rows = [list(row) for row in table.extract()]

        empty_before = _count_empty(rows)
        if empty_before and page_has_images:
            _fill_empty_cells(page, table, rows)
        empty_after = _count_empty(rows)

        populated = [row for row in rows if any((cell or "").strip() for cell in row)]
        total_cells = sum(len(row) for row in populated)
        filled = sum(1 for row in populated for cell in row if (cell or "").strip())

        if (
            filled < MIN_TABLE_FILLED_CELLS
            or total_cells == 0
            or filled / total_cells < MIN_TABLE_FILL_RATIO
        ):
            continue  # layout artefact: leave its text in the body

        markdown, n_rows, n_cols = _rows_to_markdown(rows)
        if n_rows >= MIN_TABLE_ROWS and n_cols >= MIN_TABLE_COLS:
            tables.append(
                ParsedTable(
                    page_no=page_no,
                    index=index,
                    markdown=markdown,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    empty_cells=empty_after,
                    ocr_cells_filled=empty_before - empty_after,
                )
            )
            boxes.append(fitz.Rect(table.bbox))

    return tables, boxes


def _body_text(page: fitz.Page, table_boxes: list[fitz.Rect]) -> str:
    """Page text with blocks falling inside detected tables removed."""
    parts = []
    for block in page.get_text("blocks"):
        rect = fitz.Rect(block[:4])
        centre = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
        if any(box.contains(centre) for box in table_boxes):
            continue
        parts.append(block[4])
    return "\n".join(parts).strip()


def _text_covered(page: fitz.Page, rect: fitz.Rect) -> bool:
    """True if live text blocks already occupy most of this region.

    Distinguishes a pasted image of text from a background or watermark
    sitting behind text that extracted normally.
    """
    area = rect.get_area()
    if area <= 0:
        return True
    for block in page.get_text("blocks"):
        if not block[4].strip():
            continue
        overlap = fitz.Rect(block[:4]) & rect
        if overlap.get_area() > 0.3 * area:
            return True
    return False


def _image_regions(page: fitz.Page, table_boxes: list[fitz.Rect]) -> list[fitz.Rect]:
    """Embedded images large enough to carry content and not already covered."""
    page_area = abs(page.rect.get_area()) or 1.0
    regions: list[fitz.Rect] = []

    for info in page.get_images(full=True):
        try:
            rects = page.get_image_rects(info[0])
        except Exception as exc:  # noqa: BLE001 - malformed image references are skipped
            log.debug("image_rect_unavailable", page=page.number, error=str(exc))
            continue

        for rect in rects:
            if rect.get_area() / page_area < MIN_IMAGE_AREA_RATIO:
                continue
            centre = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
            if any(box.contains(centre) for box in table_boxes):
                continue  # handled by cell-level recognition
            if _text_covered(page, rect):
                continue
            regions.append(rect)

    return regions


def _wordlike_score(text: str) -> float:
    """Common-word occurrences per 1000 characters."""
    if len(text) < WORDLIKE_MIN_CHARS:
        return float("inf")
    lowered = " " + text.lower() + " "
    hits = sum(lowered.count(word) for word in COMMON_WORDS)
    return hits / (len(text) / 1000)


def parse_pdf(path, extract_tables: bool = True) -> ParsedDocument:
    pages: list[ParsedPage] = []

    with fitz.open(path) as document:
        for page_no, page in enumerate(document, start=1):
            tables, boxes = _find_tables(page, page_no) if extract_tables else ([], [])
            body = _body_text(page, boxes)
            ocr_applied = False
            region_texts: list[str] = []

            page_text = "\n\n".join(
                segment for segment in [body, *(t.markdown for t in tables)] if segment
            )

            if len(body) < OCR_CHAR_THRESHOLD and not tables:
                recognised = _ocr_image(_render(page))
                if len(recognised) > len(body):
                    body, ocr_applied = recognised, True

            elif _wordlike_score(page_text) < MIN_WORDLIKE_SCORE:
                # The text layer is present but unusable; the rendered page is not.
                recognised = _ocr_image(_render(page))
                if _wordlike_score(recognised) >= MIN_WORDLIKE_SCORE:
                    body, ocr_applied = recognised, True
                    tables = []  # detected tables share the damaged encoding

            else:
                for rect in _image_regions(page, boxes):
                    try:
                        text = _ocr_image(_render(page, rect))
                    except Exception as exc:  # noqa: BLE001 - one region must not stop the page
                        log.debug("region_ocr_failed", page=page_no, error=str(exc))
                        continue
                    if len(text) >= MIN_REGION_OCR_CHARS:
                        region_texts.append(text)

            segments = [body, *region_texts, *(t.markdown for t in tables)]
            text = "\n\n".join(segment for segment in segments if segment)
            pages.append(
                ParsedPage(page_no, text, ocr_applied, len(region_texts), tables)
            )

    return ParsedDocument(pages)
