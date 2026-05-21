"""Shared PDF geometry helpers — detect input boxes, label them, place text.

This module is the single source of truth for the rectangle-driven flat-PDF
strategy used by BOTH ``form_parser`` (to extract fields) and
``form_writer`` (to place answers). Keeping the helpers here means the
parser and writer can't drift out of sync — every box the parser
emits is the SAME box the writer will fill, identified by the same
coordinates.

Each detected box is a ``dict`` with ``{x0, x1, top, bottom, width,
height}``. Coordinates are in PDF user-space points, with the
pdfplumber convention (origin top-left, y grows downward).

A ``BoxAnchor`` wraps the same data with the page index attached, so
the writer can re-locate the exact rectangle on the right page from
the parser's output without redoing detection. Anchors round-trip
through Postgres jsonb via ``to_dict``/``from_dict``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Tuning knobs for label clustering around each detected box.
# Kept here (not duplicated in writer / parser) so a tweak is one edit.
# ---------------------------------------------------------------------------

LABEL_ROW_BAND = 5.0       # vertical tolerance for "same row" label match
LABEL_GAP_LIMIT = 18.0     # max horizontal gap inside a single label cluster
LABEL_ABOVE_REACH = 55.0   # how far ABOVE the box to scan (multi-line wrapped labels)
LABEL_ABOVE_X_SLOP = 30.0  # horizontal slop when looking for labels ABOVE the box

# Drop a box if it has more than this many non-whitespace chars
# strictly inside its bounds — that's a label cell, not an input.
MAX_TEXT_CHARS_INSIDE_BOX = 3


# ---------------------------------------------------------------------------
# Typed anchor passed parser → filler → writer in FormField.metadata
# ---------------------------------------------------------------------------


@dataclass
class BoxAnchor:
    """Geometric pointer to one input box on one page.

    Serialised into ``FormField.metadata['box_anchor']`` by the parser and
    read back by the writer to place the answer in the exact same
    rectangle — no re-detection, no name matching.
    """

    page_idx: int       # 0-based page index in pdfplumber order
    x0: float
    top: float
    x1: float
    bottom: float
    page_width: float
    page_height: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_idx": self.page_idx,
            "x0": self.x0,
            "top": self.top,
            "x1": self.x1,
            "bottom": self.bottom,
            "page_width": self.page_width,
            "page_height": self.page_height,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BoxAnchor":
        return cls(
            page_idx=int(raw["page_idx"]),
            x0=float(raw["x0"]),
            top=float(raw["top"]),
            x1=float(raw["x1"]),
            bottom=float(raw["bottom"]),
            page_width=float(raw["page_width"]),
            page_height=float(raw["page_height"]),
        )

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.bottom - self.top


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------


def normalise_label(s: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation,
    and fold Unicode quotation marks to their ASCII equivalents.

    Used on both the FormParser-extracted field name AND the PDF word
    stream so matches survive minor punctuation differences. Federal
    PDFs almost always render U+2019 RIGHT SINGLE QUOTATION MARK in
    words like "Contractor's", while the FormParser-extracted field
    name comes through as ASCII U+0027 APOSTROPHE — without folding,
    the match misses silently.
    """
    s = s.lower()
    s = (
        s.replace("’", "'")  # right single quotation mark
        .replace("‘", "'")  # left single quotation mark
        .replace("“", '"')  # left double quotation mark
        .replace("”", '"')  # right double quotation mark
        .replace("–", "-")  # en dash
        .replace("—", "-")  # em dash
    )
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    s = s.rstrip(":*-—– ").strip()
    return s


# ---------------------------------------------------------------------------
# Box detection
# ---------------------------------------------------------------------------


def detect_input_boxes(page) -> list[dict]:
    """Detect input rectangles on one PDF page.

    Algorithm:
      1. Collect every rectangle that looks like a thin horizontal
         strip (width >= 20pt, height <= 2.5pt). These are the
         top/bottom edges of input boxes drawn by the PDF.
      2. Pair top + bottom strips whose x-ranges match (within 3pt
         slop) and whose vertical separation is 6..30pt — the
         typical interior height of an input box.
      3. Each pair → a candidate box.
      4. Dedupe overlapping boxes: when two boxes overlap >= 70% by
         area, keep the SMALLER (inner) one. PDFs frequently draw
         each input as two concentric frames (the outer being the
         cell border, the inner being the actual input area).
      5. Filter pathological detections: skip if width < 25pt or
         height < 6pt — too small to be a real input area.
      6. Drop boxes containing more than MAX_TEXT_CHARS_INSIDE_BOX
         non-whitespace chars — those are LABEL cells (in table-of-
         options grids), not input cells.

    Returns list of dicts ``{x0, x1, top, bottom, width, height}``,
    sorted top-to-bottom then left-to-right.
    """
    rects = list(getattr(page, "rects", []) or [])
    h_segs: list[dict] = []
    for r in rects:
        w = float(r["x1"]) - float(r["x0"])
        h = float(r["bottom"]) - float(r["top"])
        if w >= 20.0 and h <= 2.5:
            h_segs.append(r)
    h_segs.sort(key=lambda r: (round(float(r["top"]), 1), float(r["x0"])))

    boxes: list[dict] = []
    consumed_segs: set[int] = set()
    for i, top in enumerate(h_segs):
        if i in consumed_segs:
            continue
        for j in range(i + 1, len(h_segs)):
            if j in consumed_segs:
                continue
            bot = h_segs[j]
            if (
                abs(float(top["x0"]) - float(bot["x0"])) > 3.0
                or abs(float(top["x1"]) - float(bot["x1"])) > 3.0
            ):
                continue
            dy = float(bot["top"]) - float(top["bottom"])
            if dy < 6.0 or dy > 30.0:
                continue
            x0 = min(float(top["x0"]), float(bot["x0"]))
            x1 = max(float(top["x1"]), float(bot["x1"]))
            y_top = float(top["top"])
            y_bottom = float(bot["bottom"])
            w = x1 - x0
            h = y_bottom - y_top
            if w < 25.0 or h < 6.0:
                continue
            boxes.append({
                "x0": x0, "x1": x1,
                "top": y_top, "bottom": y_bottom,
                "width": w, "height": h,
            })
            consumed_segs.add(i)
            consumed_segs.add(j)
            break

    def _overlap_ratio(a: dict, b: dict) -> float:
        ix0 = max(a["x0"], b["x0"])
        ix1 = min(a["x1"], b["x1"])
        iy0 = max(a["top"], b["top"])
        iy1 = min(a["bottom"], b["bottom"])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        a_area = a["width"] * a["height"]
        b_area = b["width"] * b["height"]
        return inter / min(a_area, b_area)

    # Sort smaller first so the inner (true input) box is kept and
    # the outer cell border is discarded.
    boxes.sort(key=lambda b: b["width"] * b["height"])
    kept: list[dict] = []
    for cand in boxes:
        if any(_overlap_ratio(cand, k) >= 0.70 for k in kept):
            continue
        kept.append(cand)

    words = list(getattr(page, "extract_words", lambda **_: [])(
        x_tolerance=2, y_tolerance=2,
        keep_blank_chars=False, use_text_flow=True,
    ))

    def _chars_inside(box: dict) -> int:
        n = 0
        for w in words:
            wx0 = float(w["x0"])
            wx1 = float(w["x1"])
            wtop = float(w["top"])
            wbot = float(w["bottom"])
            if (
                wx0 >= box["x0"] - 1.0
                and wx1 <= box["x1"] + 1.0
                and wtop >= box["top"] - 1.0
                and wbot <= box["bottom"] + 1.0
            ):
                n += len(w["text"])
                if n > MAX_TEXT_CHARS_INSIDE_BOX:
                    return n
        return n

    input_boxes = [b for b in kept if _chars_inside(b) <= MAX_TEXT_CHARS_INSIDE_BOX]
    input_boxes.sort(key=lambda b: (round(b["top"], 1), b["x0"]))
    return input_boxes


# ---------------------------------------------------------------------------
# Label context (what to call each box)
# ---------------------------------------------------------------------------


def build_label_context(box: dict, words: list[dict]) -> str:
    """Find the label text for one box.

    Two strategies tried in order:

    Strategy 1 — SAME-ROW cluster (dominant case):
      Walk right-to-left from the box's left edge, growing a cluster
      of words whose vertical midpoint sits inside the box's row.
      Stops when the gap between the next-leftward word and the
      cluster exceeds LABEL_GAP_LIMIT (only after a colon has been
      seen — before that we tolerate larger gaps so a stray "$"
      between label colon and box doesn't terminate the cluster).

    Strategy 2 — fallback: lines ABOVE the box.
      Used when no same-row words were found (label-above-input
      layouts). Collects words in the rows above whose x-range
      overlaps the box's x-range (with LABEL_ABOVE_X_SLOP horizontal
      tolerance) up to LABEL_ABOVE_REACH above.

    Output is run through ``normalise_label`` for consistent
    downstream matching.
    """
    box_x0 = float(box["x0"])
    box_x1 = float(box["x1"])
    box_top = float(box["top"])
    box_bot = float(box["bottom"])

    same_row = []
    y_lo = box_top - LABEL_ROW_BAND
    y_hi = box_bot + LABEL_ROW_BAND
    for w in words:
        w_mid_y = (float(w["top"]) + float(w["bottom"])) / 2.0
        if w_mid_y < y_lo or w_mid_y > y_hi:
            continue
        if float(w["x1"]) > box_x0 + 2.0:
            continue
        same_row.append(w)
    same_row.sort(key=lambda w: float(w["x0"]))

    if same_row:
        rightward_first = list(reversed(same_row))
        cluster: list[dict] = []
        next_x = box_x0
        saw_colon = False
        for w in rightward_first:
            gap = next_x - float(w["x1"])
            if cluster and saw_colon and gap > LABEL_GAP_LIMIT:
                break
            cluster.insert(0, w)
            next_x = float(w["x0"])
            if w["text"].rstrip().endswith(":"):
                saw_colon = True
        if cluster:
            return normalise_label(" ".join(w["text"] for w in cluster))

    above_x_min = box_x0 - LABEL_ABOVE_X_SLOP
    above_x_max = box_x1 + LABEL_ABOVE_X_SLOP
    above_y_min = box_top - LABEL_ABOVE_REACH
    above_y_max = box_top + 1.0
    above = [
        w for w in words
        if float(w["bottom"]) <= above_y_max
        and float(w["top"]) >= above_y_min
        and float(w["x1"]) >= above_x_min
        and float(w["x0"]) <= above_x_max
    ]
    above.sort(key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))
    return normalise_label(" ".join(w["text"] for w in above))


# ---------------------------------------------------------------------------
# Text placement helpers
# ---------------------------------------------------------------------------


def truncate_to_width(text: str, max_width: float, font_size: float) -> str:
    """Truncate ``text`` so it fits in ``max_width`` at ``font_size``.

    Helvetica isn't fixed-width — avg glyph advance is roughly 0.50
    (digits) to 0.55 (letters) times font_size. 0.52 is the midpoint;
    we append "…" when truncating so the reader knows there's more.
    """
    if not text:
        return ""
    avg_char_width = font_size * 0.52
    if avg_char_width <= 0:
        return text
    max_chars = int(max_width / avg_char_width)
    if max_chars >= len(text):
        return text
    if max_chars < 4:
        return text[:max(0, max_chars)]
    return text[: max_chars - 1] + "…"


def placement_for_box(box: dict) -> dict:
    """Compute font size, baseline, and max-width for placing text inside a box.

    Used by the writer for both the legacy match-then-place flow and
    the new anchor-driven flow. Box dict needs ``x0, x1, top, bottom,
    height`` (matches ``detect_input_boxes`` output and
    ``BoxAnchor`` after the four core fields).
    """
    x = float(box["x0"]) + 4.0
    # PDF baseline is at glyph bottom; box.bottom - 3pt is a good
    # vertical centre for 10pt Helvetica.
    y_baseline_pdf = float(box["bottom"]) - 3.0
    font_size = max(7.0, min(12.0, float(box["height"]) - 4.0))
    max_width = max(20.0, float(box["x1"]) - float(box["x0"]) - 8.0)
    return {
        "x": x,
        "y_baseline_pdf": y_baseline_pdf,
        "font_size": font_size,
        "max_width": max_width,
    }
