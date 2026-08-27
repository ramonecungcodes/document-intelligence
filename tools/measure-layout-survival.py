"""Does page layout survive the degradation profiles?

The question behind the LayoutLM proposal is narrow and checkable without training
anything: a fax destroys glyphs, but does it destroy *geometry*? LayoutLM's whole
advantage is that it reads word positions as well as words, so if the positions are
noise on a fax it has nothing the text-only classifier does not already have.

Three measurements, from word boxes that already exist on disk:

  retention   how many words docTR still finds on a degraded page, against the exact
              count from the clean PDF's own text layer. Words it never found have no
              box, and layout it cannot see cannot help it.

  fidelity    cosine similarity between a degraded document's occupancy grid and its
              own clean original. High means the ink is still in the same places.

  discrimination  the one that matters. Nearest neighbour over the clean corpus by
              layout alone -- no words at all -- excluding the document's own original.
              If a fax page still retrieves its own document type from geometry, then
              geometry carries type signal the ruined text does not.

Fidelity without discrimination would be a trap: every business document is a block of
text on US Letter, so two pages can be 0.95 similar and share nothing useful. The
nearest-neighbour test is what separates "the ink is in the same places" from "the ink
being in those places says what this document is".
"""
import json, math, os, sys, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
REPO = sys.argv[1] if len(sys.argv) > 1 else "."
CACHE = os.path.join(REPO, "data", "normalized", "doctr")
CLEAN = os.path.join(REPO, "data")
DEGRADED = os.path.join(REPO, "data", "degraded")
COLS, ROWS = 12, 16

import pymupdf


def grid(words, width, height):
    """Coarse occupancy of page one: what fraction of each cell carries ink."""
    cells = [0.0] * (COLS * ROWS)
    if not width or not height:
        return cells
    for text, page, x0, y0, x1, y1 in words:
        if page != 1:
            continue
        # Spread each box over the cells it overlaps, weighted by area, so a wide
        # header contributes across the columns it actually spans.
        c0, c1 = int(x0 / width * COLS), int(x1 / width * COLS)
        r0, r1 = int(y0 / height * ROWS), int(y1 / height * ROWS)
        for r in range(max(0, r0), min(ROWS - 1, r1) + 1):
            for c in range(max(0, c0), min(COLS - 1, c1) + 1):
                cells[r * COLS + c] += 1.0
    return cells


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def page_size(path):
    doc = pymupdf.open(path)
    try:
        r = doc[0].rect
        return r.width, r.height
    finally:
        doc.close()


def native_words(path):
    doc = pymupdf.open(path)
    try:
        out = []
        for i, page in enumerate(doc, 1):
            if i > 1:
                break
            for x0, y0, x1, y1, text, *_ in page.get_text("words"):
                out.append((text, i, x0, y0, x1, y1))
        r = doc[0].rect
        return out, r.width, r.height
    finally:
        doc.close()


TYPE_OF = {"forms": "form", "invoices": "invoice",
           "multi_bill_invoices": "multi_bill_invoice",
           "purchase_orders": "purchase_order", "resumes": "resume"}

# ---------------------------------------------------------------- clean reference
print("building the clean reference set from embedded text layers ...", flush=True)
reference = {}          # relative path -> (doc_type, unit grid, word count)
for dirpath, _dirs, files in os.walk(CLEAN):
    rel_dir = os.path.relpath(dirpath, CLEAN).replace("\\", "/")
    if rel_dir.split("/")[0] in {"degraded", "normalized", "labels", "source_html", "."}:
        if rel_dir != ".":
            continue
    for name in files:
        if not name.endswith(".pdf"):
            continue
        rel = f"{rel_dir}/{name}" if rel_dir != "." else name
        words, w, h = native_words(os.path.join(dirpath, name))
        folder = rel.split("/")[0]
        if folder not in TYPE_OF:
            continue
        reference[rel] = (TYPE_OF[folder], unit(grid(words, w, h)), len(words))
print(f"  {len(reference)} clean documents", flush=True)

# ---------------------------------------------------------------- degraded probes
rows = []
for dirpath, _dirs, files in os.walk(CACHE):
    for name in sorted(files):
        if not name.endswith(".pdf.json"):
            continue
        rel = os.path.relpath(os.path.join(dirpath, name), CACHE).replace("\\", "/")
        rel = rel[:-5]                                  # drop .json
        stem = os.path.basename(rel)
        base, _, profile = stem.partition("__")
        profile = profile.replace(".pdf", "")
        origin = f"{os.path.dirname(rel)}/{base}.pdf"
        if origin not in reference:
            continue
        payload = json.load(open(os.path.join(dirpath, name), encoding="utf-8"))
        words = [(w[0], w[1], w[2], w[3], w[4], w[5]) for w in payload.get("words") or []]
        w, h = page_size(os.path.join(DEGRADED, rel))
        g = unit(grid(words, w, h))
        truth = TYPE_OF[origin.split("/")[0]]
        clean_type, clean_grid, clean_n = reference[origin]
        page1 = [x for x in words if x[1] == 1]

        # nearest neighbour over layout alone, its own original held out
        best, best_score = None, -1.0
        for other, (other_type, other_grid, _n) in reference.items():
            if other == origin:
                continue
            s = cosine(g, other_grid)
            if s > best_score:
                best, best_score = other, s
        rows.append({
            "file": rel, "profile": profile, "truth": truth,
            "retention": (len(page1) / clean_n) if clean_n else None,
            "fidelity": cosine(g, clean_grid),
            "nn_type": reference[best][0], "nn_score": best_score,
        })

# ---------------------------------------------------------------- report
by_profile = collections.defaultdict(list)
for r in rows:
    by_profile[r["profile"]].append(r)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")

print()
print("LAYOUT SURVIVAL")
print()
print(f"  {'profile':<10}{'n':>5}{'word retention':>16}{'layout fidelity':>17}"
      f"{'1-NN type acc':>15}")
for profile in ("light", "photo", "fax"):
    rs = by_profile.get(profile) or []
    if not rs:
        continue
    acc = sum(1 for r in rs if r["nn_type"] == r["truth"]) / len(rs)
    print(f"  {profile:<10}{len(rs):>5}{mean(r['retention'] for r in rs):>16.3f}"
          f"{mean(r['fidelity'] for r in rs):>17.3f}{acc:>15.3f}")
print(f"  {'majority':<10}{'':>5}{'':>16}{'':>17}{0.2:>15.3f}   <- guessing the commonest type")

print()
print("  per-type 1-NN accuracy from layout alone")
print(f"  {'type':<22}" + "".join(f"{p:>10}" for p in ("light", "photo", "fax")))
for t in sorted(TYPE_OF.values()):
    line = f"  {t:<22}"
    for profile in ("light", "photo", "fax"):
        rs = [r for r in by_profile.get(profile, []) if r["truth"] == t]
        line += f"{(sum(1 for r in rs if r['nn_type']==r['truth'])/len(rs)):>10.3f}" if rs else f"{'--':>10}"
    print(line)

out = os.path.join(REPO, "reports", "layout-survival.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rows, open(out, "w", encoding="utf-8", newline="\n"), indent=1)
print(f"\nwrote {out}")
