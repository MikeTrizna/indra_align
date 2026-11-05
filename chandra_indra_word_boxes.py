"""
Word boxes for Chandra's transcription, using Indra OCR's line boxes and
per-character x positions (CTC peaks) as the geometry.

    pip install indra-ocr rapidfuzz pillow numpy

Untested sketch: it uses Indra internals (OrtOCR.detect_all / recognize),
which may change between releases. Pin your indra-ocr version.
"""
import argparse, base64, html, json, mimetypes, os
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein
import indra_ocr

PAD = 0.35  # Indra's own padding around a character peak, in line heights


def indra_chars(image_path, conv):
    """Every recognized character on the page: (char, x, line_box).
    Spaces are kept (they help the alignment) but carry x too."""
    gray = np.asarray(Image.open(image_path).convert("L"))
    ocr = conv.ocr
    boxes, _ = ocr.detect_all(gray)  # NOTE: no deskew/rotation here
    rec = ocr.recognize(gray, boxes)
    lines = []
    for box, (text, conf, toks) in zip(boxes, rec):
        chars = []
        for tok, x in toks:
            ch = tok if isinstance(tok, str) else ocr.vocab.decode([tok])
            for c in ch:  # a multi-char token shares one x
                chars.append((c, float(x)))
        if chars:
            lines.append((list(map(float, box[:4])), chars))
    return lines


def lines_in_block(lines, bbox):
    """Indra lines whose centre falls in a Chandra block, top to bottom."""
    x0, y0, x1, y1 = bbox
    sel = [ln for ln in lines
           if x0 <= (ln[0][0] + ln[0][2]) / 2 <= x1
           and y0 <= (ln[0][1] + ln[0][3]) / 2 <= y1]
    return sorted(sel, key=lambda ln: (ln[0][1], ln[0][0]))


def align_block(chandra_text, block_lines):
    """Map Chandra words onto Indra geometry. Returns
    [{"word", "boxes": [[x0,y0,x1,y1], ...], "match": 0..1}]"""
    # Flatten Indra lines into one string, joining lines with a space.
    ref, geo = [], []  # geo[i] = (x, line_index) for ref[i]
    for li, (box, chars) in enumerate(block_lines):
        if ref:
            ref.append(" "); geo.append((None, li))
        for c, x in chars:
            ref.append(c); geo.append((x, li))
    ref_s = "".join(ref)
    hyp = " ".join(chandra_text.split())  # Chandra reflows; drop its newlines

    # hyp position -> ref position (or None), plus exact-match flag
    m = [None] * len(hyp); exact = [False] * len(hyp)
    for op in Levenshtein.opcodes(hyp, ref_s):
        if op.tag in ("equal", "replace"):
            n = min(op.src_end - op.src_start, op.dest_end - op.dest_start)
            for k in range(n):
                m[op.src_start + k] = op.dest_start + k
                exact[op.src_start + k] = op.tag == "equal"

    out, i = [], 0
    for word in hyp.split(" "):
        span = range(i, i + len(word)); i += len(word) + 1
        hits = [geo[m[j]] for j in span if m[j] is not None and geo[m[j]][0] is not None]
        if not hits:
            out.append({"word": word, "boxes": [], "match": 0.0})
            continue
        boxes = []
        for li in sorted({h[1] for h in hits}):  # a hyphenated word spans lines
            xs = [h[0] for h in hits if h[1] == li]
            bx0, by0, bx1, by1 = block_lines[li][0]
            h = max(1.0, by1 - by0)
            boxes.append([max(bx0, min(xs) - PAD * h), by0,
                          min(bx1, max(xs) + PAD * h), by1])
        out.append({"word": word, "boxes": boxes,
                    "match": sum(exact[j] for j in span) / max(1, len(word))})
    return out


def fill_gaps(words):
    """Words with no aligned characters: place them between their neighbours."""
    for k, w in enumerate(words):
        if w["boxes"]:
            continue
        prev = next((words[j]["boxes"][-1] for j in range(k - 1, -1, -1) if words[j]["boxes"]), None)
        nxt = next((words[j]["boxes"][0] for j in range(k + 1, len(words)) if words[j]["boxes"]), None)
        if prev and nxt and abs(prev[1] - nxt[1]) < 1:  # same line
            w["boxes"] = [[prev[2], prev[1], nxt[0], prev[3]]]
        w["guessed"] = True
    return words


# ---------------------------------------------------------------- outputs

def save_json(result, image_path, size, path):
    doc = {"image": str(image_path), "width": size[0], "height": size[1],
           "blocks": result}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)


def _lines(words):
    """Group a block's word boxes into lines by their (shared) line y-extent."""
    lines = {}
    for w in words:
        for b in w["boxes"]:
            key = (round(b[1]), round(b[3]))
            lines.setdefault(key, []).append((b, w))
    return [sorted(v, key=lambda t: t[0][0]) for _, v in sorted(lines.items())]


def save_hocr(result, image_path, size, path):
    """Standard hOCR: ocr_page > ocr_carea > ocr_line > ocrx_word.
    A word split across lines appears once per line piece."""
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
        for li, line in enumerate(_lines(blk["words"]), 1):
            lb = [min(b[0] for b, _ in line), min(b[1] for b, _ in line),
                  max(b[2] for b, _ in line), max(b[3] for b, _ in line)]
            out.append(f'<span class="ocr_line" id="line_{bi}_{li}" title="{bb(lb)}">')
            for b, w in line:
                wid += 1
                conf = int(round(100 * w["match"]))
                out.append(f'<span class="ocrx_word" id="word_{wid}" '
                           f'title="{bb(b)}; x_wconf {conf}">{html.escape(w["word"])}</span>')
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
 --good:#1f8a5b;--mid:#c58a12;--bad:#c2362b;--guess:#6a4fb3;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#191c1b;--panel:#222625;--ink:#e6e9e7;
 --muted:#a0a9a5;--line:#3a403e;--good:#3fbf86;--mid:#e3aa3a;--bad:#ef6a5e;--guess:#a58cf0;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.45 "Source Sans 3","Segoe UI",system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);
 display:flex;flex-wrap:wrap;gap:12px 22px;align-items:center;padding:10px 18px}
h1{font-size:15px;font-weight:600;margin:0;max-width:30ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
label{display:flex;gap:6px;align-items:center;color:var(--muted)}
input[type=range]{width:140px}
#readout{margin-left:auto;min-width:16ch;font-weight:600}
#readout small{font-weight:400;color:var(--muted)}
.legend{display:flex;gap:12px;color:var(--muted);font-size:13px}
.legend i{display:inline-block;width:12px;height:12px;border:2px solid;margin-right:4px;vertical-align:-2px}
main{padding:18px;overflow:auto}
#stage{position:relative;margin:0 auto;max-width:100%;width:__W__px;line-height:0}
#stage img{width:100%;height:auto;display:block}
.blk{position:absolute;border:1px dashed var(--muted);opacity:.55;pointer-events:none}
.w{position:absolute;border:2px solid var(--good);background:transparent;cursor:pointer;padding:0}
.w.mid{border-color:var(--mid)}.w.bad{border-color:var(--bad)}
.w.guess{border-color:var(--guess);border-style:dashed}
.w:hover,.w:focus-visible,.w.on{background:color-mix(in srgb,currentColor 22%,transparent);outline:2px solid var(--ink);outline-offset:1px}
.w.good{color:var(--good)}.w.mid{color:var(--mid)}.w.bad{color:var(--bad)}.w.guess{color:var(--guess)}
.dim{opacity:.12}
.hideblk .blk,.hidewords .w{display:none}
</style></head><body>
<header>
 <h1 title="__TITLE__">__TITLE__</h1>
 <label><input type="checkbox" id="showW" checked> Words</label>
 <label><input type="checkbox" id="showB" checked> Blocks</label>
 <label>Review below <input type="range" id="thr" min="0" max="100" value="100"> <span id="thrv">100%</span></label>
 <span class="legend"><span><i style="border-color:var(--good)"></i>agree</span>
  <span><i style="border-color:var(--mid)"></i>partial</span>
  <span><i style="border-color:var(--bad)"></i>disagree</span>
  <span><i style="border-color:var(--guess);border-style:dashed"></i>placed between neighbours</span></span>
 <span id="readout">Hover a word <small>(__N__ words)</small></span>
</header>
<main><div id="stage"><img src="__SRC__" alt="Page image"></div></main>
<script>
const W=__W__,H=__H__,DATA=__DATA__;
const stage=document.getElementById('stage'),out=document.getElementById('readout');
const pct=(v,d)=>(100*v/d)+'%';
function place(el,b){el.style.left=pct(b[0],W);el.style.top=pct(b[1],H);
 el.style.width=pct(b[2]-b[0],W);el.style.height=pct(b[3]-b[1],H);}
const words=[];
DATA.forEach(blk=>{
 const d=document.createElement('div');d.className='blk';place(d,blk.bbox);stage.appendChild(d);
 blk.words.forEach(w=>{
  const cls=w.guessed?'guess':w.match>=.999?'good':w.match>=.5?'mid':'bad';
  w.boxes.forEach(b=>{
   const e=document.createElement('button');e.className='w '+cls;place(e,b);
   e.setAttribute('aria-label',w.word+', '+Math.round(100*w.match)+'% agreement');
   e.dataset.m=w.match;
   const show=()=>{out.innerHTML='';const s=document.createElement('span');s.textContent=w.word;
    const m=document.createElement('small');m.textContent='  '+Math.round(100*w.match)+'% agreement'+(w.guessed?', position guessed':'');
    out.append(s,m);};
   e.addEventListener('mouseenter',show);e.addEventListener('focus',show);
   e.addEventListener('click',()=>{words.forEach(x=>x.classList.remove('on'));e.classList.add('on');show();});
   stage.appendChild(e);words.push(e);});});});
const thr=document.getElementById('thr'),thrv=document.getElementById('thrv');
thr.addEventListener('input',()=>{const t=thr.value/100;thrv.textContent=thr.value+'%';
 words.forEach(e=>e.classList.toggle('dim',t<1&&+e.dataset.m>=t));});
document.getElementById('showW').addEventListener('change',e=>stage.classList.toggle('hidewords',!e.target.checked));
document.getElementById('showB').addEventListener('change',e=>stage.classList.toggle('hideblk',!e.target.checked));
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Word boxes for Chandra text via Indra geometry")
    ap.add_argument("image")
    ap.add_argument("blocks", help='JSON: [{"bbox": [x0,y0,x1,y1] in pixels, "text": "..."}]')
    ap.add_argument("-o", "--out", help="word-box JSON (default: <image>.words.json)")
    ap.add_argument("--html", help="write an HTML overlay viewer to this path")
    ap.add_argument("--hocr", help="write hOCR to this path")
    ap.add_argument("--no-embed", action="store_true",
                    help="HTML links to the image instead of embedding it")
    a = ap.parse_args()

    chandra_blocks = json.load(open(a.blocks, encoding="utf-8"))
    size = Image.open(a.image).size
    conv = indra_ocr.load(threads=4, mode="ocr")
    lines = indra_chars(a.image, conv)
    result = []
    for blk in chandra_blocks:
        words = align_block(blk["text"], lines_in_block(lines, blk["bbox"]))
        result.append({"bbox": blk["bbox"], "words": fill_gaps(words)})

    out = a.out or str(Path(a.image).with_suffix("")) + ".words.json"
    save_json(result, a.image, size, out)
    print("wrote", out)
    if a.html:
        save_html(result, a.image, size, a.html, embed=not a.no_embed)
        print("wrote", a.html)
    if a.hocr:
        save_hocr(result, a.image, size, a.hocr)
        print("wrote", a.hocr)