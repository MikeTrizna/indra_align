#!/usr/bin/env python3
"""
Run Indra OCR's full-page pipeline on one image and write tabular output:

  <prefix>.lines.csv   one row per line, in Indra's reading order
  <prefix>.chars.csv   one row per character: centroid position + its line_id
  <prefix>.page.json   page metadata (rotation, skew, size) and layout blocks

    pip install indra-ocr==1.1.2 pillow numpy

Why not just use the public API? `OrtOCR.__call__` returns lines in reading
order, but throws away the per-character positions. This script wraps two
internal functions (`tidy_lines`, `reading_order`) during a single full-page
call to capture the final lines together with their character tokens, so the
text, reading order and character positions all come from the same pass.
It depends on indra-ocr internals: pin the version.

IDs and order
-------------
`line_id` is a stable identifier (Indra's internal line index after it
splits/merges lines); it says nothing about order and is the join key between
lines.csv and chars.csv. `reading_order` (0 = read first) is the line's
position in Indra's reading order, and lines.csv rows are written in that
order. `block_id` indexes the blocks in page.json, which are also in reading
order.

Coordinates
-----------
Everything is in pixels of the input image (origin top-left). Indra only
gives an x per character (the CTC peak), so `cy` is the vertical middle of
the line's box, not a true glyph centroid. If Indra turned the page by a
quarter turn (`rotation` != 0 in page.json), coordinates are on the turned
page; the small deskew is always undone.

`char_idx` indexes into the line's `text` (spaces included), so
text[char_idx] == char whenever the line's tokens decode to exactly its text
(a warning is printed for lines where they don't).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import indra_ocr
from indra_ocr import pipeline as P

TESTED_VERSION = "1.1.2"


def run_page(conv, gray):
    """Full-page OCR. Returns (page, recs, boxes, order, frame_wh) where recs /
    boxes are Indra's final lines (text, conf, tokens) / [x0,y0,x1,y1] in the
    deskewed frame, and order[k] is the index of page.lines[k] in them."""
    ocr = conv.ocr
    cap = {}

    orig_tidy = P.tidy_lines
    orig_order = ocr.reading_order
    orig_detect = ocr.detect_all

    def spy_tidy(boxes, rec, *a, **kw):
        b, r = orig_tidy(boxes, rec, *a, **kw)
        cap["boxes"], cap["rec"] = np.array(b, float), list(r)
        return b, r

    def spy_order(*a, **kw):
        out = orig_order(*a, **kw)
        cap["order"] = list(out[0])  # last call wins (tables re-run it)
        return out

    def spy_detect(g, *a, **kw):
        cap["frame_wh"] = (g.shape[1], g.shape[0])  # turned + deskewed page
        return orig_detect(g, *a, **kw)

    P.tidy_lines = spy_tidy
    ocr.reading_order = spy_order
    ocr.detect_all = spy_detect
    try:
        page = ocr(gray)
    finally:
        P.tidy_lines = orig_tidy
        del ocr.reading_order, ocr.detect_all  # drop the instance overrides

    missing = {"boxes", "rec", "order", "frame_wh"} - cap.keys()
    if missing:
        sys.exit(f"Could not capture Indra internals ({sorted(missing)}); "
                 f"this script was written for indra-ocr {TESTED_VERSION}.")
    if len(cap["order"]) != len(page.lines) or len(cap["rec"]) != len(page.lines):
        sys.exit("Captured lines do not match the page's lines (table cells "
                 "were probably re-split); unsupported with this indra-ocr "
                 f"version. Tested with {TESTED_VERSION}.")
    for k, i in enumerate(cap["order"]):
        if cap["rec"][i][0] != page.lines[k].text:
            sys.exit(f"Captured text for line {k} differs from Indra's page "
                     f"line; this script was written for indra-ocr "
                     f"{TESTED_VERSION}.")
    return page, cap["rec"], cap["boxes"], cap["order"], cap["frame_wh"]


def to_page_frame(x, y, page, frame_wh):
    """Map a point from the deskewed frame back to the page as Indra reports
    its line boxes (mirrors what OrtOCR._read does to the line boxes)."""
    if page.skew:
        x, y = P.unrotate_box([x, y, x, y], page.skew, frame_wh)[:2]
    if page.vertical:  # (x, y) -> (W - y, x), W = turned page height
        x, y = frame_wh[1] - y, x
    return x, y


def block_for(box, blocks):
    """Index of the smallest non-picture block containing the line's centre."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best, best_area = None, None
    for j, b in enumerate(blocks):
        x0, y0, x1, y1 = b["bbox"]
        if b["label"] != "Picture" and x0 <= cx <= x1 and y0 <= cy <= y1:
            area = (x1 - x0) * (y1 - y0)
            if best is None or area < best_area:
                best, best_area = j, area
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="page image (png, jpg, tiff, ...)")
    ap.add_argument("-o", "--output-prefix", default=None,
                    help="output path prefix (default: next to the image, "
                         "same stem)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if indra_ocr.__version__ != TESTED_VERSION:
        print(f"[warn] written for indra-ocr {TESTED_VERSION}, found "
              f"{indra_ocr.__version__}", file=sys.stderr)

    img = Image.open(args.image)
    gray = np.asarray(img.convert("L"))
    conv = indra_ocr.load(threads=args.threads, mode="ocr")
    ocr = conv.ocr

    page, recs, boxes, order, frame_wh = run_page(conv, gray)

    # ---- lines.csv
    line_rows, char_rows, bad = [], [], 0
    for reading_order, i in enumerate(order):
        line_id = i  # stable id: Indra's internal line index, not the order
        ln = page.lines[reading_order]
        text, conf, toks = recs[i]
        block_id = block_for(ln.box, page.blocks)
        line_rows.append({
            "line_id": line_id,
            "reading_order": reading_order,
            "block_id": "" if block_id is None else block_id,
            "block_label": "" if block_id is None else page.blocks[block_id]["label"],
            "text": ln.text,
            "x0": ln.box[0], "y0": ln.box[1], "x1": ln.box[2], "y1": ln.box[3],
            "rec_conf": ln.conf,
        })

        # ---- chars.csv: one row per character; a multi-character token
        # shares its single x position.
        cy_deskew = float(boxes[i][1] + boxes[i][3]) / 2
        chars = []
        for tok, x in (toks or []):
            s = tok if isinstance(tok, str) else ocr.vocab.decode([tok])
            for c in s:
                chars.append((c, float(x)))
        if "".join(c for c, _ in chars) != ln.text:
            bad += 1
        for idx, (c, x) in enumerate(chars):
            cx, cy = to_page_frame(x, cy_deskew, page, frame_wh)
            char_rows.append({"line_id": line_id, "char_idx": idx, "char": c,
                              "cx": round(cx, 1), "cy": round(cy, 1)})
    if bad:
        print(f"[warn] {bad} line(s) whose tokens do not decode to exactly "
              f"the line text; char_idx is the token-sequence position there",
              file=sys.stderr)

    prefix = Path(args.output_prefix) if args.output_prefix \
        else Path(args.image).with_suffix("")
    for suffix, rows in ((".lines.csv", line_rows), (".chars.csv", char_rows)):
        path = prefix.with_name(prefix.name + suffix)
        fields = list(rows[0]) if rows else (
            ["line_id", "reading_order", "block_id", "block_label", "text", "x0", "y0", "x1",
             "y1", "rec_conf"] if suffix == ".lines.csv"
            else ["line_id", "char_idx", "char", "cx", "cy"])
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[info] wrote {path} ({len(rows)} rows)", file=sys.stderr)

    # ---- page.json
    meta = {
        "image": str(args.image),
        "width": img.width,
        "height": img.height,
        "rotation": page.rotation,      # degrees CCW the page was turned
        "skew": page.skew,              # degrees of tilt undone
        "vertical": page.vertical,
        "source": page.source,
        "note": page.note,
        "n_lines": len(page.lines),
        "n_blocks": len(page.blocks),
        "blocks": [{"block_id": j, "label": b["label"], "bbox": b["bbox"],
                    "text": b["text"]} for j, b in enumerate(page.blocks)],
        "indra_ocr_version": indra_ocr.__version__,
    }
    path = prefix.with_name(prefix.name + ".page.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"[info] wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
