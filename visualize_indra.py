#!/usr/bin/env python3
"""Build a stand-alone HTML viewer for Indra OCR results.

Takes the output of indra_page_data.py (<prefix>.page.json, <prefix>.lines.csv,
<prefix>.chars.csv) plus the page image, and writes one self-contained HTML file
(image embedded) with toggleable layers for blocks, lines and character dots.

    python visualize_indra.py newspaper_example/sn83016063_1801-11-30_1_1.jpg \
        newspaper_example/indra_results -o newspaper_example/newspaper_example_indra.html

The second argument may be the prefix or any of the three files.
"""
import argparse
import base64
import csv
import json
import mimetypes
import sys
from pathlib import Path

SUFFIXES = (".page.json", ".lines.csv", ".chars.csv")


def resolve_prefix(arg: str) -> Path:
    for s in SUFFIXES:
        if arg.endswith(s):
            return Path(arg[: -len(s)])
    return Path(arg)


def load(prefix: Path):
    page = json.loads(Path(f"{prefix}.page.json").read_text())
    lines = {}
    with open(f"{prefix}.lines.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lid = int(r["line_id"])
            lines[lid] = {
                "id": lid,
                "order": int(r["reading_order"]),
                "block": int(r["block_id"]) if r["block_id"] != "" else -1,
                "label": r["block_label"],
                "text": r["text"],
                "box": [float(r[k]) for k in ("x0", "y0", "x1", "y1")],
                "conf": float(r["rec_conf"]) if r["rec_conf"] else None,
                "chars": [],
            }
    with open(f"{prefix}.chars.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ln = lines.get(int(r["line_id"]))
            if ln is not None:
                ln["chars"].append([round(float(r["cx"]), 1), round(float(r["cy"]), 1)])
    return page, sorted(lines.values(), key=lambda l: l["order"])


def build(image_path: Path, prefix: Path, embed: bool = True) -> str:
    page, lines = load(prefix)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    if embed:
        src = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode()
    else:
        src = image_path.name
    data = {
        "width": page["width"],
        "height": page["height"],
        "meta": {k: page.get(k) for k in ("rotation", "skew", "vertical", "source", "note", "indra_ocr_version")},
        "blocks": page["blocks"],
        "lines": lines,
    }
    html = TEMPLATE.replace("__TITLE__", image_path.name)
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    html = html.replace("__IMG__", src)
    return html


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("results", help="indra_page_data.py output prefix (or one of its files)")
    ap.add_argument("-o", "--output", help="output HTML (default: <image stem>.indra.html)")
    ap.add_argument("--no-embed", action="store_true", help="link to the image by file name instead of embedding it")
    a = ap.parse_args()

    img = Path(a.image)
    out = Path(a.output) if a.output else img.with_suffix(".indra.html")
    prefix = resolve_prefix(a.results)
    html = build(img, prefix, embed=not a.no_embed)
    if a.no_embed:
        html = html.replace(f'src="{img.name}"', f'src="{Path(img).resolve().relative_to(out.resolve().parent) if out.resolve().parent in img.resolve().parents else img.resolve()}"')
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Indra OCR – __TITLE__</title>
<style>
:root{--bg:#1d1f23;--panel:#272a30;--fg:#e6e6e6;--mut:#9aa0a8;--bd:#3a3e46}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;overflow:hidden}
#bar{position:fixed;inset:0 0 auto 0;z-index:5;display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--bd)}
#bar label{display:flex;align-items:center;gap:5px;cursor:pointer;user-select:none}
#bar input[type=range]{width:110px}
#bar button{background:var(--bg);color:var(--fg);border:1px solid var(--bd);border-radius:4px;padding:3px 9px;cursor:pointer}
#bar .sep{width:1px;height:18px;background:var(--bd)}
#bar .mut{color:var(--mut)}
#view{position:fixed;inset:44px 0 0 0;overflow:hidden;cursor:grab;touch-action:none}
#view.drag{cursor:grabbing}
#stage{position:absolute;left:0;top:0;transform-origin:0 0}
#stage img{display:block;user-select:none;-webkit-user-drag:none;transition:opacity .1s}
#stage svg{position:absolute;left:0;top:0;overflow:visible}
.blk{fill-opacity:.10;stroke-width:2;vector-effect:non-scaling-stroke}
.blk:hover{fill-opacity:.25}
.ln{fill:none;stroke-width:1.2;vector-effect:non-scaling-stroke}
.lnhit{fill:transparent;stroke:none}
.lnhit:hover{fill:rgba(255,255,255,.18)}
.sel{fill:rgba(255,230,0,.22)!important;stroke:#ffe600!important}
#tip{position:fixed;z-index:9;pointer-events:none;max-width:420px;background:rgba(15,16,18,.95);border:1px solid var(--bd);border-radius:5px;padding:6px 9px;display:none;white-space:pre-wrap}
#tip b{color:#fff}
#tip .m{color:var(--mut);font-size:11px}
#legend{display:flex;gap:10px;flex-wrap:wrap}
#legend span{display:flex;align-items:center;gap:4px}
#legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
</style></head><body>
<div id="bar">
 <label><input type="checkbox" id="tImg" checked> Image</label>
 <label><input type="checkbox" id="tBlk" checked> Blocks</label>
 <label><input type="checkbox" id="tLn" checked> Lines</label>
 <label><input type="checkbox" id="tCh"> Character dots</label>
 <label class="mut">Dot size <input type="range" id="dot" min="1" max="12" value="4"></label>
 <label class="mut">Image dim <input type="range" id="dim" min="0" max="100" value="0"></label>
 <span class="sep"></span>
 <label class="mut">Line color
  <select id="lc"><option value="conf">by confidence</option><option value="block">by block type</option></select></label>
 <span class="sep"></span>
 <button id="fit">Fit</button><button id="one">100%</button>
 <span class="sep"></span>
 <span id="legend"></span>
 <span class="mut" id="stats"></span>
</div>
<div id="view"><div id="stage"><img id="img" alt="page" src="__IMG__"><svg id="svg"></svg></div></div>
<div id="tip"></div>
<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const PAL = {"Text":"#4aa3ff","Section-Header":"#ff7a45","Page-Header":"#b37feb","Page-Footer":"#b37feb","List-Group":"#36cfc9","Table":"#73d13d","Picture":"#f759ab","Figure":"#f759ab","Caption":"#ffc53d","Footnote":"#ffc53d","Equation-Block":"#95de64"};
const EXTRA = ["#ff85c0","#ffd666","#5cdbd3","#adc6ff","#d3adf7","#ffa39e"];
function col(l){ if(!(l in PAL)) PAL[l]=EXTRA[Object.keys(PAL).length % EXTRA.length]; return PAL[l]; }
const $ = id => document.getElementById(id);
const W = D.width, H = D.height;
const svg = $("svg"), stage = $("stage"), view = $("view"), img = $("img"), tip = $("tip");
svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
img.width = W; img.height = H;
function g(parent){ const e=document.createElementNS(NS,"g"); parent.appendChild(e); return e; }
function el(parent, tag, attrs){ const e=document.createElementNS(NS,tag); for(const k in attrs) e.setAttribute(k,attrs[k]); parent.appendChild(e); return e; }
const gBlk=g(svg), gLn=g(svg), gHit=g(svg), gCh=g(svg);
const blockById = {}; D.blocks.forEach(b => blockById[b.block_id]=b);
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

// ---- blocks
D.blocks.forEach(b => {
  const [x0,y0,x1,y1]=b.bbox, c=col(b.label);
  const r = el(gBlk,"rect",{x:x0,y:y0,width:x1-x0,height:y1-y0,fill:c,stroke:c,class:"blk"});
  r._b = b;
});
// ---- lines
function confColor(c){ if(c==null) return "#888"; const t=Math.max(0,Math.min(1,(c-0.3)/0.6)); return `hsl(${Math.round(t*120)},85%,55%)`; }
const lnEls = [];
D.lines.forEach(l => {
  const [x0,y0,x1,y1]=l.box;
  const r = el(gLn,"rect",{x:x0,y:y0,width:x1-x0,height:y1-y0,class:"ln"});
  const h = el(gHit,"rect",{x:x0,y:y0,width:x1-x0,height:y1-y0,class:"lnhit"});
  h._l = l; lnEls.push([l,r]);
});
function recolor(){
  const byConf = $("lc").value==="conf";
  lnEls.forEach(([l,r]) => r.setAttribute("stroke", byConf ? confColor(l.conf) : col(l.label)));
  const lg = $("legend");
  lg.innerHTML = byConf
    ? `<span class="mut">line conf</span><span><i style="background:${confColor(0.3)}"></i>low</span><span><i style="background:${confColor(0.6)}"></i></span><span><i style="background:${confColor(0.9)}"></i>high</span>`
    : "";
  const labels=[...new Set(D.blocks.map(b=>b.label))];
  lg.innerHTML += `<span class="mut">blocks</span>` + labels.map(l=>`<span><i style="background:${col(l)}"></i>${l}</span>`).join("");
}
$("lc").onchange = recolor; recolor();

// ---- character dots (built lazily)
let dotsBuilt=false;
function buildDots(){
  if(dotsBuilt) return; dotsBuilt=true;
  const f=document.createDocumentFragment();
  D.lines.forEach(l => {
    const c = confColor(l.conf);
    l.chars.forEach(([cx,cy]) => {
      const e=document.createElementNS(NS,"circle");
      e.setAttribute("cx",cx); e.setAttribute("cy",cy); e.setAttribute("fill",c);
      f.appendChild(e);
    });
  });
  gCh.appendChild(f); setDot();
}
function setDot(){ const r=+$("dot").value * 0.8 / scale; gCh.querySelectorAll("circle").forEach(c=>c.setAttribute("r",r)); }
// strokes are non-scaling; dots rescale on zoom (debounced)
let dotTimer; function dotsSoon(){ clearTimeout(dotTimer); dotTimer=setTimeout(()=>{ if($("tCh").checked) setDot(); },80); }

// ---- toggles
const vis = (id, node) => { node.style.display = $(id).checked ? "" : "none"; };
function applyToggles(){
  vis("tBlk",gBlk); vis("tLn",gLn); vis("tCh",gCh);
  gHit.style.display = ($("tLn").checked || $("tBlk").checked) ? "" : "none";
  img.style.visibility = $("tImg").checked ? "visible" : "hidden";
  img.style.opacity = 1 - $("dim").value/100;
  if($("tCh").checked) buildDots();
  gBlk.style.pointerEvents = "none";
}
["tImg","tBlk","tLn","tCh","dim"].forEach(i => $(i).addEventListener("input", applyToggles));
$("dot").addEventListener("input", setDot);
$("stats").textContent = `${D.blocks.length} blocks · ${D.lines.length} lines · ${D.lines.reduce((a,l)=>a+l.chars.length,0)} chars`;
applyToggles();

// ---- hover tip
let selEl=null;
gHit.addEventListener("mousemove", e => {
  const l = e.target._l; if(!l) return;
  const b = blockById[l.block];
  tip.innerHTML = `<div class="m">line ${l.id} · order ${l.order} · block ${l.block} (${l.label}) · conf ${l.conf==null?"–":l.conf.toFixed(2)} · ${l.chars.length} chars</div><b>${esc(l.text)||"<i>(empty)</i>"}</b>`;
  tip.style.display="block";
  const x=Math.min(e.clientX+14, innerWidth-tip.offsetWidth-8), y=Math.min(e.clientY+14, innerHeight-tip.offsetHeight-8);
  tip.style.left=x+"px"; tip.style.top=y+"px";
});
gHit.addEventListener("mouseleave", ()=>tip.style.display="none");

// ---- pan / zoom
let scale=1, tx=0, ty=0;
function apply(){ stage.style.transform=`translate(${tx}px,${ty}px) scale(${scale})`; dotsSoon(); }
function fit(){
  const vw=view.clientWidth, vh=view.clientHeight;
  scale=Math.min(vw/W, vh/H); tx=(vw-W*scale)/2; ty=(vh-H*scale)/2; apply();
}
function zoomAt(px,py,f){
  const ns=Math.max(0.02,Math.min(8,scale*f)); f=ns/scale;
  tx=px-(px-tx)*f; ty=py-(py-ty)*f; scale=ns; apply();
}
view.addEventListener("wheel", e => { e.preventDefault(); const r=view.getBoundingClientRect(); zoomAt(e.clientX-r.left,e.clientY-r.top,Math.exp(-e.deltaY*(e.ctrlKey?0.01:0.0015))); }, {passive:false});
let drag=null, moved=0;
view.addEventListener("pointerdown", e => { drag={x:e.clientX,y:e.clientY,tx,ty}; moved=0; view.setPointerCapture(e.pointerId); view.classList.add("drag"); });
view.addEventListener("pointermove", e => { if(!drag) return; tx=drag.tx+e.clientX-drag.x; ty=drag.ty+e.clientY-drag.y; moved++; apply(); tip.style.display="none"; });
view.addEventListener("pointerup", e => { drag=null; view.classList.remove("drag"); });
$("fit").onclick=fit; $("one").onclick=()=>{ const r=view.getBoundingClientRect(); zoomAt(r.width/2,r.height/2,1/scale); };
addEventListener("resize", fit);
addEventListener("keydown", e => { if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT") return;
  if(e.key==="b") $("tBlk").click(); if(e.key==="l") $("tLn").click(); if(e.key==="c") $("tCh").click(); if(e.key==="i") $("tImg").click(); if(e.key==="f") fit(); });
fit();
</script></body></html>
"""

if __name__ == "__main__":
    main()
