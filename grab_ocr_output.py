#!/usr/bin/env python3
"""
Get block-level OCR (text + bounding boxes) for one page of a NARA
Revolutionary War pension file, using two Revolution Crossroads datasets:

  1. RevolutionCrossroads/nara_revolutionary_war_pension_files
     -> maps the NARA catalog URL (NAID + objectPage) to the file's page list,
        so we can find the page's zero-based position in the processed PDF.
  2. RevolutionCrossroads/RevX_OCR_results (block_level config)
     -> the Chandra OCR blocks for that page, with bboxes.

Usage:
    pip install duckdb
    python get_page_ocr.py "https://catalog.archives.gov/id/196376039?objectPage=63" -o page_63.json

Output:
    [
      {"bbox": [112, 340, 1460, 820], "text": "Know all men by these presents ..."},
      ...
    ]

Bboxes are [x1, y1, x2, y2] in the pixel space of the page as it was
processed for OCR (some PDFs were resized), which may not match the
original NARA JPG exactly.
"""

import argparse
import json
import os
import re
import sys
from urllib.parse import parse_qs, urlparse

import duckdb

PENSION_REPO = "RevolutionCrossroads/nara_revolutionary_war_pension_files"
OCR_REPO = "RevolutionCrossroads/RevX_OCR_results"


def hf_glob(repo, path, revision=None):
    rev = f"@{revision}" if revision else ""
    return f"hf://datasets/{repo}{rev}/{path}"


def parse_catalog_url(url):
    """Return (naid, object_page) from a catalog.archives.gov URL."""
    parsed = urlparse(url)
    m = re.search(r"/id/(\d+)", parsed.path)
    if not m:
        raise ValueError(f"Could not find a NAID in URL: {url}")
    naid = m.group(1)
    qs = parse_qs(parsed.query)
    object_page = int(qs.get("objectPage", ["1"])[0])
    return naid, object_page


def connect():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    token = os.environ.get("HF_TOKEN")
    if token:  # optional; avoids rate limits on large scans
        con.execute("CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN ?)", [token])
    return con


def find_page_index(con, naid, object_page, revision=None):
    """
    Look the file up in the pension dataset and return
    (zero_based_page_index, page_row_dict_or_None).
    The OCR page_number is the page's position in the assembled PDF, which
    is the order of pageNumber within the NAID.
    """
    src = hf_glob(PENSION_REPO, "data/*.parquet", revision)
    rows = con.execute(
        f"""
        SELECT pageNumber, pageObjectId, pageImageURL, RevX_OCR_text
        FROM read_parquet('{src}')
        WHERE NAID = ?
        ORDER BY pageNumber
        """,
        [naid],
    ).fetchall()

    if not rows:
        print(f"[warn] NAID {naid} not found in {PENSION_REPO}; "
              f"assuming page index = objectPage - 1", file=sys.stderr)
        return object_page - 1, None

    page_numbers = [r[0] for r in rows]
    if object_page not in page_numbers:
        raise ValueError(
            f"objectPage {object_page} not in NAID {naid} "
            f"(pages {page_numbers[0]}..{page_numbers[-1]}, {len(rows)} total)"
        )
    idx = page_numbers.index(object_page)
    r = rows[idx]
    info = {"pageNumber": r[0], "pageObjectId": r[1],
            "pageImageURL": r[2], "RevX_OCR_text": r[3]}
    if idx != object_page - 1:
        print(f"[info] file has gaps in page numbering; objectPage "
              f"{object_page} is processed page index {idx}", file=sys.stderr)
    return idx, info


def block_sort_key(block_id):
    """'/page/62/Text/7' -> 7 (the numeric suffix the dataset uses for order)."""
    m = re.search(r"(\d+)\s*$", block_id or "")
    return int(m.group(1)) if m else 0


def get_blocks(con, naid, page_index, revision=None):
    src = hf_glob(OCR_REPO, "block_level/*.parquet", revision)
    rows = con.execute(
        f"""
        SELECT block_id, block_type, block_bbox, block_text, OCR_run_ID
        FROM read_parquet('{src}')
        WHERE collection = 'rev_war_pension'
          AND filename = ?
          AND page_number = ?
        """,
        [f"{naid}.pdf", page_index],
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="catalog.archives.gov URL with ?objectPage=N")
    ap.add_argument("-o", "--output", help="write JSON here (default: stdout)")
    ap.add_argument("--ocr-revision", default=None,
                    help="pin RevX_OCR_results to a tag, e.g. new_OCR_v2")
    ap.add_argument("--pension-revision", default=None,
                    help="pin the pension dataset to a tag/commit")
    args = ap.parse_args()

    naid, object_page = parse_catalog_url(args.url)
    print(f"[info] NAID={naid} objectPage={object_page}", file=sys.stderr)

    con = connect()
    page_index, page_info = find_page_index(con, naid, object_page,
                                            args.pension_revision)
    blocks, run_id = get_blocks(con, naid, page_index, args.ocr_revision)

    if not blocks:
        sys.exit(f"No OCR blocks found for {naid}.pdf page_number={page_index}")
    print(f"[info] {len(blocks)} blocks from {run_id} "
          f"(page_number={page_index})", file=sys.stderr)

    # Sanity check against the page-level OCR text stored in the pension dataset.
    if page_info and page_info.get("RevX_OCR_text"):
        joined = "\n".join(b["text"] for b in blocks)
        if norm(joined) != norm(page_info["RevX_OCR_text"]):
            print("[warn] block text differs from the pension dataset's "
                  "RevX_OCR_text for this page (possibly a different OCR run "
                  "or page alignment) - check against the image:\n  "
                  f"{page_info.get('pageImageURL')}", file=sys.stderr)

    out = json.dumps(blocks, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"[info] wrote {args.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()