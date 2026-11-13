#!/usr/bin/env python3
"""
Get block-level OCR (text + bounding boxes) for one page, using the
RevolutionCrossroads/RevX_OCR_results dataset.

The page is identified by the same two keys used in the dataset's
"page_level" subset: `filename` and `page_number`. The page_level subset only
has plain text, so the bboxes come from the "block_level" subset, which uses
the same keys. The page_level text is used as a sanity check.

Usage:
    pip install duckdb
    python grab_ocr_output.py 196376039.pdf 62 -o page_63.json

Output:
    [
      {"bbox": [112, 340, 1460, 820], "text": "Know all men by these presents ..."},
      ...
    ]

Bboxes are [x1, y1, x2, y2] in the pixel space of the page as it was
processed for OCR (some PDFs were resized), which may not match the
original image exactly.
"""

import argparse
import json
import os
import re
import sys

import duckdb

OCR_REPO = "RevolutionCrossroads/RevX_OCR_results"


def hf_glob(repo, path, revision=None):
    rev = f"@{revision}" if revision else ""
    return f"hf://datasets/{repo}{rev}/{path}"


def connect():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    token = os.environ.get("HF_TOKEN")
    if token:  # optional; avoids rate limits on large scans
        con.execute("CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN ?)", [token])
    return con


def block_sort_key(block_id):
    """'/page/62/Text/7' -> 7 (the numeric suffix the dataset uses for order)."""
    m = re.search(r"(\d+)\s*$", block_id or "")
    return int(m.group(1)) if m else 0


def get_blocks(con, filename, page_number, revision=None, collection=None):
    src = hf_glob(OCR_REPO, "block_level/*.parquet", revision)
    rows = con.execute(
        f"""
        SELECT block_id, block_type, block_bbox, block_text, OCR_run_ID
        FROM read_parquet('{src}')
        WHERE filename = ?
          AND page_number = ?
          AND (? IS NULL OR collection = ?)
        """,
        [filename, page_number, collection, collection],
    ).fetchall()

    if not rows:
        return [], None

    # If the page was processed in more than one run, keep the latest one.
    latest_run = max(r[4] for r in rows)
    rows = [r for r in rows if r[4] == latest_run]
    rows.sort(key=lambda r: block_sort_key(r[0]))

    blocks = []
    for block_id, block_type, bbox_str, text, _ in rows:
        text = (text or "").strip()
        if not text:
            continue  # skip pictures / empty blocks
        bbox = [int(round(v)) for v in json.loads(bbox_str)]
        blocks.append({"bbox": bbox, "text": text})
    return blocks, latest_run


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def get_page_text(con, filename, page_number, run_id, revision=None, collection=None):
    """Plain page text from the page_level subset (same OCR run), or None."""
    src = hf_glob(OCR_REPO, "page_level/*.parquet", revision)
    row = con.execute(
        f"""
        SELECT new_ocr_text
        FROM read_parquet('{src}')
        WHERE filename = ? AND page_number = ? AND OCR_run_ID = ?
          AND (? IS NULL OR collection = ?)
        LIMIT 1
        """,
        [filename, page_number, run_id, collection, collection],
    ).fetchone()
    return row[0] if row else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filename", help="dataset `filename` value, e.g. 196376039.pdf")
    ap.add_argument("page_number", type=int, help="dataset `page_number` value")
    ap.add_argument("-o", "--output", help="write JSON here (default: stdout)")
    ap.add_argument("--collection", default=None,
                    help="restrict to a collection (e.g. rev_war_pension) if the "
                         "filename is ambiguous")
    ap.add_argument("--ocr-revision", default=None,
                    help="pin RevX_OCR_results to a tag, e.g. new_OCR_v2")
    args = ap.parse_args()

    con = connect()
    blocks, run_id = get_blocks(con, args.filename, args.page_number,
                                args.ocr_revision, args.collection)

    if not blocks:
        sys.exit(f"No OCR blocks found for {args.filename} "
                 f"page_number={args.page_number}")
    print(f"[info] {len(blocks)} blocks from {run_id} "
          f"({args.filename} page_number={args.page_number})", file=sys.stderr)

    # Sanity check against the page_level subset.
    page_text = get_page_text(con, args.filename, args.page_number, run_id,
                              args.ocr_revision, args.collection)
    if page_text:
        joined = "\n".join(b["text"] for b in blocks)
        if norm(joined) != norm(page_text):
            print("[warn] block text differs from the page_level text for "
                  "this page - check against the image", file=sys.stderr)

    out = json.dumps(blocks, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"[info] wrote {args.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
