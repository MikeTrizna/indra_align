#!/usr/bin/env python3
"""
Word boxes for Chandra's transcription, using the line-level output of
indra_page_data.py (indra_results.lines.csv / .chars.csv / .page.json) as the
geometry.

    pip install rapidfuzz numpy        # no Indra install needed at this stage

Pipeline
--------
  1. Scale.   Chandra's block boxes are in the pixel space of the page *as it
              was sent to Chandra*, which can differ from the image (some pages
              were resized). The scale is estimated by maximising how much of
              the Indra text lines fall inside the scaled blocks (override with
              --scale).
  2. Blocks.  Every Indra *character* is assigned to the Chandra block that
              contains its (x, y) position. A line that runs across two columns
              is therefore split between the blocks instead of being given
              wholesale to one of them. Within a block the pieces are ordered
              top to bottom.
  3. Lines.   Chandra's words are cut into consecutive chunks, one per Indra
              line, by a dynamic program that minimises the total edit
              distance between each chunk and its line's text. Because the cut
              points are word boundaries and the line order is fixed, one
              garbled or missing Indra line cannot drag the rest of the block
              out of alignment (the failure mode of a single page-long edit
              alignment). Chandra words that no line can plausibly explain are
              left without a box and flagged.
  4. Chars.   Inside each (chunk, line) pair a small character alignment maps
              Chandra characters to Indra character x positions. Characters
              Indra got wrong are filled in by interpolation between their
              aligned neighbours, so every placed word gets a box.
  5. Boxes.   A word's box runs from half a character pitch before its first
              character to half a pitch after its last, over the line's y
              extent; neighbouring boxes are kept from overlapping.

`match` (0..1) is the fraction of the word's characters that agree exactly with
Indra: a rough confidence, and a pointer to words worth reviewing.
"""
import argparse, base64, csv, html, json, mimetypes, os
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

BEAM = 80          # edit-cost slack kept when searching line/word cut points
SKIP = 0.7         # cost of leaving a char unexplained, vs 1.0 for a mismatch:
                   # text with <30% similarity is skipped, not force-fitted
TOL_LH = 0.25      # a char may sit this many line heights outside a block


# ------------------------------------------------------------------ loading

def load_indra(prefix):
    """-> (lines, page_meta). lines: [{id, order, text, box, chars:[(c,x,y)]}]"""
    prefix = str(prefix)
    with open(prefix + ".lines.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_line = defaultdict(list)
    with open(prefix + ".chars.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_line[int(r["line_id"])].append(
                (int(r["char_idx"]), r["char"], float(r["cx"]), float(r["cy"])))
    lines = []
    for r in rows:
        lid = int(r["line_id"])
        chars = [(c, x, y) for _, c, x, y in sorted(by_line.get(lid, []))]
        lines.append({"id": lid, "order": int(r["reading_order"]),
                      "text": r["text"],
                      "box": [float(r[k]) for k in ("x0", "y0", "x1", "y1")],
                      "chars": chars})
    meta = {}
    if os.path.exists(prefix + ".page.json"):
        with open(prefix + ".page.json", encoding="utf-8") as f:
            meta = json.load(f)
    return lines, meta


def image_size(image, meta):
    if meta.get("width") and meta.get("height"):
        return meta["width"], meta["height"]
    from PIL import Image
    return Image.open(image).size


# -------------------------------------------------------------------- scale

def estimate_scale(lines, blocks):
    """Scale s such that Chandra bbox * s lands on the Indra lines."""
    L = np.array([l["box"] for l in lines if l["text"].strip()], float)
    B = np.array([b["bbox"] for b in blocks], float)
    if not len(L) or not len(B):
        return 1.0
    area = (L[:, 2] - L[:, 0]) * (L[:, 3] - L[:, 1])
    area = np.maximum(area, 1.0)[:, None]

    def score(s):
        ix = np.clip(np.minimum(L[:, None, 2], B[None, :, 2] * s)
                     - np.maximum(L[:, None, 0], B[None, :, 0] * s), 0, None)
        iy = np.clip(np.minimum(L[:, None, 3], B[None, :, 3] * s)
                     - np.maximum(L[:, None, 1], B[None, :, 1] * s), 0, None)
        return float((ix * iy / area).sum())

    grid = np.arange(0.3, 3.0, 0.01)
    best = max(grid, key=score)
    fine = np.arange(best - 0.01, best + 0.0101, 0.001)
    best = max(fine, key=score)
    if score(1.0) >= 0.98 * score(best):
        return 1.0
    return float(best)


# ----------------------------------------------------- chars -> blocks/lines

def split_into_blocks(lines, blocks):
    """Give each Indra character to the block that contains it.
    Returns per block a list of pieces {id, text, xs, box, order}, top to
    bottom. blocks' bbox must already be in image pixels."""
    pieces = [defaultdict(list) for _ in blocks]  # block -> line id -> chars
    bb = np.array([b["bbox"] for b in blocks], float).reshape(-1, 4)
    line_by_id = {l["id"]: l for l in lines}
    for ln in lines:
        lh = max(1.0, ln["box"][3] - ln["box"][1])
        tol = TOL_LH * lh
        for ch in ln["chars"]:
            _, x, y = ch
            if not len(bb):
                break
            margin = np.minimum.reduce([x - bb[:, 0], bb[:, 2] - x,
                                        y - bb[:, 1], bb[:, 3] - y])
            j = int(np.argmax(margin))
            if margin[j] >= -tol:
                pieces[j][ln["id"]].append(ch)
    out = []
    for per_line in pieces:
        ps = []
        for lid, chs in per_line.items():
            ln = line_by_id[lid]
            text = "".join(c for c, _, _ in chs)
            if not text.strip():
                continue
            xs = np.maximum.accumulate(np.array([x for _, x, _ in chs], float))
            ps.append({"id": lid, "order": ln["order"], "text": text, "xs": xs,
                       "y0": ln["box"][1], "y1": ln["box"][3],
                       "cy": (ln["box"][1] + ln["box"][3]) / 2})
        ps.sort(key=lambda p: (p["cy"], p["xs"][0]))
        out.append(ps)
    return out


# --------------------------------------------------- stage 1: cut into lines

def cut_words_into_lines(words, texts):
    """Assign consecutive runs of `words` to `texts` (one per Indra line)
    minimising total edit distance. Returns (chunks, skipped) where chunks[l]
    is (j, k): words[j:k] belong to line l, and skipped is a set of word
    indices that no line explains."""
    n, L = len(words), len(texts)
    INF = float("inf")
    skip_cost = [SKIP * (len(w) + 1) for w in words]
    # prefix char offsets of the single-space join, to bound chunk length
    off = [0]
    for w in words:
        off.append(off[-1] + len(w) + 1)

    f = [dict() for _ in range(L + 1)]   # f[l][j]: best cost, l lines / j words used
    back = [dict() for _ in range(L + 1)]
    f[0][0] = 0.0
    for l in range(L + 1):
        row = f[l]
        # skipping words (left box-less) before line l
        for j in range(n):
            if j in row:
                c = row[j] + skip_cost[j]
                if c < row.get(j + 1, INF):
                    row[j + 1] = c
                    back[l][j + 1] = ("skip", j)
        if l == L or not row:
            continue
        best = min(row.values())
        text = texts[l]
        cap = 2 * len(text) + 12
        for j, c0 in list(row.items()):
            if c0 > best + BEAM:
                continue
            # empty chunk: this Indra line has no Chandra counterpart
            c = c0 + SKIP * len(text)
            if c < f[l + 1].get(j, INF):
                f[l + 1][j] = c
                back[l + 1][j] = ("line", j, j)
            for k in range(j + 1, n + 1):
                if off[k] - off[j] - 1 > cap:
                    break
                chunk = " ".join(words[j:k])
                c = c0 + Levenshtein.distance(chunk, text)
                if c < f[l + 1].get(k, INF):
                    f[l + 1][k] = c
                    back[l + 1][k] = ("line", j, k)
    # trailing words nothing could explain
    chunks, skipped = [(0, 0)] * L, set()
    if n not in f[L]:
        # beam pruned the exact end: finish by skipping from the best state
        j = min(f[L], key=lambda jj: f[L][jj] + sum(skip_cost[jj:]))
        skipped.update(range(j, n))
        n_end = j
    else:
        n_end = n
    l, j = L, n_end
    while l > 0 or j > 0:
        kind = back[l][j]
        if kind[0] == "skip":
            skipped.add(kind[1]); j = kind[1]
        else:
            _, a, b = kind
            chunks[l - 1] = (a, b); l -= 1; j = a
    return chunks, skipped


# ------------------------------------------------ stage 2: chars inside line

def place_chunk(chunk_words, piece):
    """Boxes for the words of one chunk on one Indra line piece.
    Returns [(box, match, guessed)] aligned with chunk_words."""
    text, xs = piece["text"], piece["xs"]
    s = " ".join(chunk_words)
    n_ref = len(text)
    pitch = (xs[-1] - xs[0]) / (n_ref - 1) if n_ref > 1 else 0.0
    if pitch <= 0:
        pitch = max(1.0, (piece["y1"] - piece["y0"]) * 0.45)
    xl, xr = xs[0] - pitch / 2, xs[-1] + pitch / 2

    pos = np.full(len(s), np.nan)       # x per chunk char (anchored)
    exact = np.zeros(len(s), bool)
    for op in Levenshtein.opcodes(s, text):
        if op.tag not in ("equal", "replace"):
            continue
        a, b = op.src_end - op.src_start, op.dest_end - op.dest_start
        for k in range(a):
            d = op.dest_start + (k if op.tag == "equal" else int(k * b / a))
            pos[op.src_start + k] = xs[min(d, n_ref - 1)]
            exact[op.src_start + k] = op.tag == "equal"
    anchored = ~np.isnan(pos)
    idx = np.arange(len(s))
    if anchored.any():
        ai = idx[anchored]
        full = np.interp(idx, ai, pos[anchored])
        # beyond the outermost anchors, extend at the line's char pitch
        full[idx < ai[0]] = pos[ai[0]] - pitch * (ai[0] - idx[idx < ai[0]])
        full[idx > ai[-1]] = pos[ai[-1]] + pitch * (idx[idx > ai[-1]] - ai[-1])
    else:  # nothing agreed: spread evenly over the line
        full = np.linspace(xs[0], xs[-1], len(s))
    full = np.clip(np.maximum.accumulate(full), xl + pitch / 2, xr - pitch / 2)

    out, i = [], 0
    for w in chunk_words:
        a, b = i, i + len(w) - 1
        i += len(w) + 1
        out.append([[full[a] - pitch / 2, piece["y0"], full[b] + pitch / 2, piece["y1"]],
                    float(exact[a:b + 1].sum()) / max(1, len(w)),
                    not anchored[a:b + 1].any()])
    for p, q in zip(out, out[1:]):      # keep neighbours from overlapping
        if p[0][2] > q[0][0]:
            mid = (p[0][2] + q[0][0]) / 2
            p[0][2] = q[0][0] = mid
    for o in out:
        o[0][0] = max(o[0][0], xl)
        o[0][2] = min(o[0][2], xr)
        o[0][2] = max(o[0][2], o[0][0] + 1)
    return out


def align_block(text, pieces):
    """-> (words, lines) for one Chandra block."""
    words_s = text.split()
    words = [{"word": w, "boxes": [], "match": 0.0, "guessed": True, "line": None}
             for w in words_s]
    if not pieces:
        return words, []
    chunks, _ = cut_words_into_lines(words_s, [p["text"] for p in pieces])
    lines = []
    for li, (piece, (a, b)) in enumerate(zip(pieces, chunks)):
        lines.append({"id": piece["id"], "text": piece["text"],
                      "box": [float(piece["xs"][0]), piece["y0"],
                              float(piece["xs"][-1]), piece["y1"]]})
        if a == b:
            continue
        for k, (box, match, guessed) in zip(range(a, b),
                                            place_chunk(words_s[a:b], piece)):
            words[k].update(boxes=[[round(v, 1) for v in box]],
                            match=round(match, 3), guessed=guessed, line=li)
    return words, lines


# ---------------------------------------------------------------- outputs

def save_json(result, image_path, size, scale, path):
    doc = {"image": str(image_path), "width": size[0], "height": size[1],
           "chandra_scale": scale, "blocks": result}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)


def save_hocr(result, image_path, size, path):
    """Standard hOCR: ocr_page > ocr_carea > ocr_line > ocrx_word."""
    W, H = size
    bb = lambda b: "bbox %d %d %d %d" % tuple(int(round(v)) for v in b)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml"><head>',
           '<meta charset="utf-8"/><meta name="ocr-system" content="chandra+indra-align"/>',
           '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_line ocrx_word"/>',
           '</head><body>',
           f'<div class="ocr_page" id="page_1" title="image {html.escape(str(image_path))}; bbox 0 0 {W} {H}">']
    wid = 0
    for bi, blk in enumerate(result, 1):
        out.append(f'<div class="ocr_carea" id="block_{bi}" title="{bb(blk["bbox"])}">')
        for li, ln in enumerate(blk["lines"]):
            ws = [w for w in blk["words"] if w["line"] == li and w["boxes"]]
            if not ws:
                continue
            out.append(f'<span class="ocr_line" id="line_{bi}_{li + 1}" title="{bb(ln["box"])}">')
            for w in ws:
                wid += 1
                conf = int(round(100 * w["match"]))
                out.append(f'<span class="ocrx_word" id="word_{wid}" '
                           f'title="{bb(w["boxes"][0])}; x_wconf {conf}">{html.escape(w["word"])}</span>')
            out.append('</span>')
        out.append('</div>')
    out += ['</div></body></html>']
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


VIEWER = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Word boxes: __TITLE__</title>
<style>
:root{--bg:#f3f4f2;--panel:#ffffff;--ink:#1d2321;--muted:#5d6763;--line:#d5d9d6;
 --good:#1f8a5b;--mid:#c58a12;--bad:#c2362b;--guess:#6a4fb3;--il:#2a6fb0;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#191c1b;--panel:#222625;--ink:#e6e9e7;
 --muted:#a0a9a5;--line:#3a403e;--good:#3fbf86;--mid:#e3aa3a;--bad:#ef6a5e;--guess:#a58cf0;--il:#6fb0ee;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.45 "Source Sans 3","Segoe UI",system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);
 display:flex;flex-wrap:wrap;gap:8px 22px;align-items:center;padding:10px 18px}
h1{font-size:15px;font-weight:600;margin:0;max-width:30ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
label{display:flex;gap:6px;align-items:center;color:var(--muted)}
input[type=range]{width:140px}
#readout{flex-basis:100%;min-height:2.9em;font-weight:600}
#readout small{display:block;font-weight:400;color:var(--muted);font-family:ui-monospace,monospace;font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:12px;color:var(--muted);font-size:13px}
.legend i{display:inline-block;width:12px;height:12px;border:2px solid;margin-right:4px;vertical-align:-2px}
main{padding:18px;overflow:auto}
#stage{position:relative;margin:0 auto;max-width:100%;width:__W__px;line-height:0}
#stage img{width:100%;height:auto;display:block}
.blk{position:absolute;border:1px dashed var(--muted);opacity:.55;pointer-events:none}
.il{position:absolute;border-bottom:2px solid var(--il);background:color-mix(in srgb,var(--il) 8%,transparent);pointer-events:none}
.w{position:absolute;border:2px solid var(--good);background:transparent;cursor:pointer;padding:0}
.w.mid{border-color:var(--mid)}.w.bad{border-color:var(--bad)}
.w.guess{border-color:var(--guess);border-style:dashed}
.w:hover,.w:focus-visible,.w.on{background:color-mix(in srgb,currentColor 22%,transparent);outline:2px solid var(--ink);outline-offset:1px}
.w.good{color:var(--good)}.w.mid{color:var(--mid)}.w.bad{color:var(--bad)}.w.guess{color:var(--guess)}
.dim{opacity:.12}
.hideblk .blk,.hidewords .w,.hideil .il{display:none}
</style></head><body>
<header>
 <h1 title="__TITLE__">__TITLE__</h1>
 <label><input type="checkbox" id="showW" checked> Words</label>
 <label><input type="checkbox" id="showB" checked> Blocks</label>
 <label><input type="checkbox" id="showL"> Indra lines</label>
 <label>Review below <input type="range" id="thr" min="0" max="100" value="100"> <span id="thrv">100%</span></label>
 <span class="legend"><span><i style="border-color:var(--good)"></i>agree</span>
  <span><i style="border-color:var(--mid)"></i>partial</span>
  <span><i style="border-color:var(--bad)"></i>disagree</span>
  <span><i style="border-color:var(--guess);border-style:dashed"></i>position interpolated or unplaced</span></span>
 <div id="readout">Hover a word <small>(__N__ words)</small></div>
</header>
<main><div id="stage" class="hideil"><img src="__SRC__" alt="Page image"></div></main>
<script>
const W=__W__,H=__H__,DATA=__DATA__;
const stage=document.getElementById('stage'),out=document.getElementById('readout');
const pct=(v,d)=>(100*v/d)+'%';
function place(el,b){el.style.left=pct(b[0],W);el.style.top=pct(b[1],H);
 el.style.width=pct(b[2]-b[0],W);el.style.height=pct(b[3]-b[1],H);}
const words=[];
DATA.forEach(blk=>{
 const d=document.createElement('div');d.className='blk';place(d,blk.bbox);stage.appendChild(d);
 blk.lines.forEach(l=>{const e=document.createElement('div');e.className='il';place(e,l.box);stage.appendChild(e);});
 blk.words.forEach(w=>{
  const cls=w.guessed?'guess':w.match>=.999?'good':w.match>=.5?'mid':'bad';
  const ln=w.line==null?null:blk.lines[w.line];
  const show=()=>{out.innerHTML='';const s=document.createElement('span');s.textContent=w.word;
   const m=document.createElement('small');
   m.textContent=Math.round(100*w.match)+'% agreement'+(w.guessed?(w.boxes.length?', position interpolated':', not placed'):'')
    +(ln?'  |  Indra line: '+ln.text:'');
   out.append(s,m);};
  w.boxes.forEach(b=>{
   const e=document.createElement('button');e.className='w '+cls;place(e,b);
   e.setAttribute('aria-label',w.word+', '+Math.round(100*w.match)+'% agreement');
   e.dataset.m=w.match;
   e.addEventListener('mouseenter',show);e.addEventListener('focus',show);
   e.addEventListener('click',()=>{words.forEach(x=>x.classList.remove('on'));e.classList.add('on');show();});
   stage.appendChild(e);words.push(e);});});});
const thr=document.getElementById('thr'),thrv=document.getElementById('thrv');
thr.addEventListener('input',()=>{const t=thr.value/100;thrv.textContent=thr.value+'%';
 words.forEach(e=>e.classList.toggle('dim',t<1&&+e.dataset.m>=t));});
document.getElementById('showW').addEventListener('change',e=>stage.classList.toggle('hidewords',!e.target.checked));
document.getElementById('showB').addEventListener('change',e=>stage.classList.toggle('hideblk',!e.target.checked));
document.getElementById('showL').addEventListener('change',e=>stage.classList.toggle('hideil',!e.target.checked));
</script></body></html>"""


def save_html(result, image_path, size, path, embed=True):
    W, H = size
    if embed:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        src = f"data:{mime};base64," + base64.b64encode(Path(image_path).read_bytes()).decode()
    else:  # reference the image relative to the HTML file
        src = os.path.relpath(Path(image_path).resolve(), Path(path).resolve().parent)
    n = sum(len(b["words"]) for b in result)
    data = json.dumps(result, ensure_ascii=False).replace("</", "<\\/")
    page = (VIEWER.replace("__TITLE__", html.escape(Path(image_path).name))
            .replace("__SRC__", html.escape(src, quote=True))
            .replace("__W__", str(W)).replace("__H__", str(H))
            .replace("__N__", str(n)).replace("__DATA__", data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Align Chandra block text to Indra line/char geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("blocks", help='JSON: [{"bbox": [x0,y0,x1,y1], "text": "..."}]')
    ap.add_argument("--indra", help="prefix of indra_page_data.py output "
                    "(<prefix>.lines.csv, .chars.csv, .page.json); default: "
                    "indra_results next to the blocks file")
    ap.add_argument("--scale", type=float,
                    help="multiply Chandra bboxes by this to get image pixels "
                         "(default: estimated)")
    ap.add_argument("-o", "--out", help="word-box JSON (default: <image>.words.json)")
    ap.add_argument("--html", help="write an HTML overlay viewer to this path")
    ap.add_argument("--hocr", help="write hOCR to this path")
    ap.add_argument("--no-embed", action="store_true",
                    help="HTML links to the image instead of embedding it")
    a = ap.parse_args()

    prefix = a.indra or Path(a.blocks).resolve().parent / "indra_results"
    lines, meta = load_indra(prefix)
    size = image_size(a.image, meta)
    with open(a.blocks, encoding="utf-8") as f:
        chandra = json.load(f)

    scale = a.scale if a.scale else estimate_scale(lines, chandra)
    print(f"Chandra -> image scale: {scale:.3f}"
          + ("" if a.scale else " (estimated)"))
    blocks = [{"bbox": [v * scale for v in b["bbox"]], "text": b["text"]}
              for b in chandra]
    pieces = split_into_blocks(lines, blocks)

    result = []
    for blk, ps in zip(blocks, pieces):
        words, blines = align_block(blk["text"], ps)
        result.append({"bbox": [round(v, 1) for v in blk["bbox"]],
                       "words": words, "lines": blines})

    allw = [w for b in result for w in b["words"]]
    placed = [w for w in allw if w["boxes"]]
    print(f"{len(allw)} words: {len(placed)} placed, "
          f"{sum(1 for w in placed if w['guessed'])} interpolated, "
          f"{len(allw) - len(placed)} unplaced; "
          f"mean match {np.mean([w['match'] for w in placed]) if placed else 0:.2f}")

    if placed and np.mean([w["match"] for w in placed]) < 0.4:
        print("[warn] very low agreement with Indra: the Chandra blocks may "
              "belong to a different page than this image (check the page "
              "index), or the scale is wrong (try --scale)")

    out = a.out or str(Path(a.image).with_suffix("")) + ".words.json"
    save_json(result, a.image, size, scale, out)
    print("wrote", out)
    if a.html:
        save_html(result, a.image, size, a.html, embed=not a.no_embed)
        print("wrote", a.html)
    if a.hocr:
        save_hocr(result, a.image, size, a.hocr)
        print("wrote", a.hocr)


if __name__ == "__main__":
    main()
