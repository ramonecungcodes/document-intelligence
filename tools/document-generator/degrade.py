#!/usr/bin/env python3
"""Degrade rendered documents into realistic scans, photocopies and phone photos.

The generated PDFs carry a perfect text layer, so an extractor never has to touch
OCR and accuracy comes out flattering. This rasterises them and puts the image back
through a print/scan pipeline, producing image-only documents with NO text layer --
which forces the OCR path and makes the numbers honest.

Ground truth is preserved: every degraded document keeps the label of the document
it came from, `irregularities` included, so the clean set and the defective set
both degrade with no re-labelling. Each label gains a `degradation` block, so
accuracy can be reported per profile instead of as one blended number.

Usage:
  python degrade.py                                  # ../ -> ../degraded/ (medium)
  python degrade.py --levels light,medium,heavy      # three variants per document
  python degrade.py --out ../irregular               # degrade the defective set
  python degrade.py --only invoices --levels photo   # phone-photo invoices only
  python degrade.py --format png                     # images instead of image-only PDFs
  python degrade.py --engine builtin                 # Pillow-only, no augraphy needed

Profiles:
  light   crisp office scanner, 200 dpi         medium  typical 150 dpi scan
  heavy   bad scan: dirty drum, low ink, skew    photo   phone snap, warped, shadowed
  fax     photocopied twice then faxed           mixed   pick one at random per document

Requires: pymupdf, pillow, augraphy (default engine).
  pip install pymupdf pillow augraphy
The `builtin` engine needs only pymupdf + pillow, at some loss of realism.
"""
import os, io, json, random, argparse

# augraphy's numba-cached kernels try to write into site-packages and fall over on
# some installs. JIT off costs about 0.8s/page and always works; --numba opts back in.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

CATS = ["invoices", "purchase_orders", "resumes", "forms", "multi_bill_invoices"]

# dpi, geometry and the JPEG round-trip are engine-independent. `warp` is applied with
# Pillow in both engines -- augraphy has no true perspective transform, and a phone
# photo without one does not look like a phone photo.
PROFILES = {
    "light":  dict(dpi=200, skew=0.5, warp=0.0000, jpeg=(80, 92)),
    "medium": dict(dpi=150, skew=1.3, warp=0.0015, jpeg=(55, 70)),
    "heavy":  dict(dpi=120, skew=2.6, warp=0.0040, jpeg=(30, 45)),
    "photo":  dict(dpi=150, skew=1.4, warp=0.0110, jpeg=(48, 65)),
    "fax":    dict(dpi=170, skew=1.8, warp=0.0020, jpeg=(35, 50)),
}


def _need(mod, hint):
    try:
        return __import__(mod)
    except ImportError:
        raise SystemExit(f"degrade.py needs {mod!r}. Install with:  pip install {hint}")


def _pymupdf():
    """`fitz` is the old name and warns on import; prefer `pymupdf` where present."""
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        pass
    try:
        import fitz
        return fitz
    except ImportError:
        raise SystemExit("degrade.py needs PyMuPDF. Install with:  pip install pymupdf")


# ------------------------------------------------------------------ perspective
def _solve(m, b):
    """Gaussian elimination with partial pivoting, so the builtin engine needs no numpy."""
    n = len(b)
    a = [row[:] + [b[i]] for i, row in enumerate(m)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-12:
            raise ValueError("singular perspective matrix")
        a[c], a[p] = a[p], a[c]
        pv = a[c][c]
        a[c] = [v / pv for v in a[c]]
        for r in range(n):
            if r != c and a[r][c]:
                f = a[r][c]
                a[r] = [v - f * w for v, w in zip(a[r], a[c])]
    return [a[i][n] for i in range(n)]


def _persp_coeffs(dst, src):
    """Coefficients for Image.PERSPECTIVE, which maps output coords back to input."""
    m, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return _solve(m, b)


def _geom(img, prof, rng):
    """Perspective and skew, both in Pillow.

    augraphy's Geometric only accepts whole-degree rotation, and a page fed 0.6deg
    crooked is the common case worth modelling -- so geometry is done here for both
    engines, which also keeps the two visually comparable.
    """
    from PIL import Image
    w, h = img.size
    amount = prof["warp"]
    if amount > 0:
        j = lambda: rng.uniform(-amount, amount)
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [(w * j(), h * j()), (w * (1 + j()), h * j()),
               (w * (1 + j()), h * (1 + j())), (w * j(), h * (1 + j()))]
        try:
            img = img.transform((w, h), Image.PERSPECTIVE, _persp_coeffs(dst, src),
                                Image.BICUBIC, fillcolor=(255, 255, 255))
        except ValueError:
            pass
    ang = rng.uniform(-prof["skew"], prof["skew"])
    if abs(ang) > 0.02:
        img = img.rotate(ang, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return img


# ------------------------------------------------------------------ augraphy engine
def _ag_phases(prof, rng):
    """Per-profile augraphy pipeline. Returns (ink, paper, post, op_names).

    BadPhotoCopy's noise_type is pinned away from 4: that is augraphy's Worley-noise
    path, and in 8.2.6 it computes its point count as a float and then hands it to
    range() (noisegenerator.py:428). With the default noise_type=-1 it is drawn at
    random, so roughly one document in five would die. The other types look the same.
    """
    from augraphy import (InkBleed, LowInkRandomLines, LowInkPeriodicLines,
                          SubtleNoise, BrightnessTexturize, NoiseTexturize, ColorPaper,
                          BadPhotoCopy, DirtyDrum, DirtyRollers, Faxify,
                          Jpeg, LightingGradient, ShadowCast, Brightness, OneOf)
    p = PROFILES[prof]
    jpg = Jpeg(quality_range=p["jpeg"])

    if prof == "light":
        ink = [InkBleed(intensity_range=(0.1, 0.2), severity=(0.1, 0.2))]
        paper = [SubtleNoise(subtle_range=6)]
        post = [Brightness(brightness_range=(0.95, 1.05), numba_jit=0), jpg]

    elif prof == "medium":
        ink = [InkBleed(intensity_range=(0.2, 0.4), severity=(0.1, 0.2)),
               OneOf([LowInkRandomLines(count_range=(2, 5), noise_probability=0.05),
                      LowInkPeriodicLines(count_range=(1, 3), period_range=(14, 30))])]
        paper = [SubtleNoise(subtle_range=8),
                 BrightnessTexturize(texturize_range=(0.90, 0.99), deviation=0.04)]
        post = [DirtyDrum(line_width_range=(1, 2), line_concentration=0.003,
                          noise_intensity=0.02, noise_value=(180, 220)),
                LightingGradient(numba_jit=0), SubtleNoise(subtle_range=6), jpg]

    elif prof == "heavy":
        ink = [InkBleed(intensity_range=(0.4, 0.6), severity=(0.2, 0.35)),
               LowInkRandomLines(count_range=(4, 10), noise_probability=0.1)]
        paper = [NoiseTexturize(sigma_range=(3, 8), turbulence_range=(2, 3)),
                 BrightnessTexturize(texturize_range=(0.82, 0.96), deviation=0.08)]
        post = [BadPhotoCopy(noise_type=rng.choice([1, 2, 3, 5]),
                             noise_value=(150, 220), noise_sparsity=(0.7, 0.95),
                             noise_concentration=(0.02, 0.10)),
                DirtyDrum(line_width_range=(1, 3), line_concentration=0.008,
                          noise_intensity=0.06, noise_value=(140, 200)),
                LightingGradient(numba_jit=0), jpg]

    elif prof == "photo":
        ink = [InkBleed(intensity_range=(0.2, 0.4), severity=(0.1, 0.3))]
        paper = [SubtleNoise(subtle_range=8),
                 ColorPaper(hue_range=(20, 45), saturation_range=(4, 18))]
        post = [ShadowCast(shadow_opacity_range=(0.25, 0.6), shadow_width_range=(0.4, 0.9),
                           shadow_height_range=(0.4, 0.9)),
                LightingGradient(mode="gaussian", numba_jit=0),
                SubtleNoise(subtle_range=10), jpg]

    else:  # fax
        ink = [InkBleed(intensity_range=(0.5, 0.7), severity=(0.3, 0.45)),
               LowInkPeriodicLines(count_range=(1, 2), period_range=(20, 34))]
        paper = [SubtleNoise(subtle_range=6)]
        post = [DirtyRollers(line_width_range=(24, 40), numba_jit=0),
                Faxify(monochrome=1, numba_jit=0), jpg]

    ops = (["PerspectiveWarp"] if p["warp"] else []) + ["Skew"] + \
          [type(x).__name__ for x in ink + paper + post]
    return ink, paper, post, ops


def _augraphy(pages, prof, rng):
    _need("augraphy", "augraphy")
    _need("numpy", "numpy")
    import numpy as np
    from augraphy import AugraphyPipeline
    from PIL import Image
    p = PROFILES[prof]
    ink, paper, post, ops = _ag_phases(prof, rng)
    out = []
    for i, img in enumerate(pages):
        img = _geom(img, p, rng)
        pipe = AugraphyPipeline(ink_phase=ink, paper_phase=paper, post_phase=post,
                                random_seed=rng.randrange(2 ** 31))
        res = pipe(np.array(img.convert("RGB")))
        arr = res["output"] if isinstance(res, dict) else res
        out.append(Image.fromarray(arr).convert("RGB"))
    return out, ops


# ------------------------------------------------------------------ builtin engine
PAPER_TINTS = [None, (252, 251, 246), (250, 250, 252), (253, 250, 244), (248, 248, 245)]
DESK_TONES = [(72, 68, 62), (96, 92, 88), (54, 56, 60), (120, 112, 100)]
BUILTIN = {
    "light":  dict(blur=0.40, noise=6,  dust=0.00004, grad=0.10, gray=0, bilevel=0, bleed=0.00, contrast=0.97),
    "medium": dict(blur=0.70, noise=12, dust=0.00016, grad=0.22, gray=0, bilevel=0, bleed=0.35, contrast=0.92),
    "heavy":  dict(blur=1.15, noise=20, dust=0.00045, grad=0.34, gray=1, bilevel=0, bleed=0.70, contrast=0.84),
    "photo":  dict(blur=0.90, noise=14, dust=0.00006, grad=0.40, gray=0, bilevel=0, bleed=0.20, contrast=0.90,
                   shadow=1, bg=1),
    "fax":    dict(blur=0.90, noise=16, dust=0.00090, grad=0.18, gray=1, bilevel=128, bleed=0.90, contrast=0.80),
}


def _builtin(pages, prof, rng):
    """Pillow-only fallback. Coarser than augraphy but needs no heavy dependencies."""
    from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter
    p, b = PROFILES[prof], BUILTIN[prof]
    out = []
    for img in pages:
        if b["bleed"] and rng.random() < 0.85:                      # toner spread
            img = Image.blend(img, img.filter(ImageFilter.MinFilter(3)),
                              b["bleed"] * rng.uniform(0.5, 1.0))
        img = ImageEnhance.Contrast(img).enhance(b["contrast"] + rng.uniform(-0.03, 0.03))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.97, 1.04))
        tint = rng.choice(PAPER_TINTS)                              # scanner white point
        if tint:
            img = Image.blend(img, Image.new("RGB", img.size, tint), 0.20)
        img = _geom(img, p, rng)
        w, h = img.size
        g = (Image.linear_gradient("L")
             .rotate(rng.uniform(0, 360), resample=Image.BICUBIC, fillcolor=128)
             .resize((w, h), Image.BICUBIC))
        lo = int(255 * (1.0 - b["grad"]))
        img = ImageChops.multiply(img, g.point(lambda v, lo=lo: lo + v * (255 - lo) // 255)
                                  .convert("RGB"))
        if b.get("shadow") and rng.random() < 0.7:                  # a hand over the page
            sh = Image.new("L", (w, h), 255)
            x0, y0 = rng.uniform(-0.3, 0.75) * w, rng.uniform(-0.3, 0.75) * h
            ImageDraw.Draw(sh).ellipse(
                [x0, y0, x0 + w * rng.uniform(0.5, 1.1), y0 + h * rng.uniform(0.4, 0.9)],
                fill=rng.randint(150, 205))
            img = ImageChops.multiply(img, sh.filter(
                ImageFilter.GaussianBlur(max(w, h) * 0.06)).convert("RGB"))
        if b["blur"] > 0.05:
            img = img.filter(ImageFilter.GaussianBlur(b["blur"] * rng.uniform(0.6, 1.2)))
        if b["noise"]:
            n = Image.effect_noise((w, h), b["noise"] * rng.uniform(0.7, 1.2)).convert("RGB")
            img = ImageChops.add(img, n, scale=1, offset=-128)
        specks = int(w * h * b["dust"] * rng.uniform(0.6, 1.4))     # dust on the platen
        if specks:
            d = ImageDraw.Draw(img)
            for _ in range(specks):
                x, y, r = rng.randrange(w), rng.randrange(h), rng.choice([0, 0, 0, 1, 1, 2])
                v = rng.randint(0, 90) if rng.random() < 0.75 else rng.randint(225, 255)
                d.ellipse([x - r, y - r, x + r, y + r], fill=(v, v, v))
        if b.get("bg"):                                             # page sits on a desk
            pad = int(min(w, h) * rng.uniform(0.02, 0.07))
            canvas = Image.new("RGB", (w + pad * 2, h + pad * 2), rng.choice(DESK_TONES))
            canvas = ImageChops.add(canvas, Image.effect_noise(canvas.size, 9).convert("RGB"),
                                    scale=1, offset=-128)
            canvas.paste(img, (pad, pad)); img = canvas
        if b["gray"] or b["bilevel"]:
            img = img.convert("L")
            if b["bilevel"]:
                t = b["bilevel"] + rng.randint(-18, 18)
                img = img.point(lambda v, t=t: 255 if v > t else 0)
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(*p["jpeg"]), subsampling=2)
        buf.seek(0)
        out.append(Image.open(buf).convert("RGB"))
    ops = ["ink_bleed", "paper_tint"] + (["perspective"] if p["warp"] else []) + \
          ["skew", "uneven_light"] + (["shadow"] if b.get("shadow") else []) + \
          ["blur", "grain", "dust"] + (["desk_background"] if b.get("bg") else []) + \
          (["grayscale"] if b["gray"] else []) + (["bilevel"] if b["bilevel"] else []) + ["jpeg"]
    return out, ops


# ------------------------------------------------------------------ io
def _rasterize(fitz, path, dpi):
    from PIL import Image
    doc = fitz.open(path)
    try:
        pages = []
        for pg in doc:
            pm = pg.get_pixmap(dpi=dpi, alpha=False)
            pages.append(Image.frombytes("RGB", (pm.width, pm.height), pm.samples))
        return pages
    finally:
        doc.close()


def _write(pages, base, fmt, dpi):
    """Image-only PDF (no text layer) by default; png/jpg for eyeballing."""
    if fmt == "pdf":
        pages[0].save(base + ".pdf", "PDF", resolution=float(dpi),
                      save_all=True, append_images=pages[1:])
        return base + ".pdf"
    ext = "." + fmt
    if len(pages) == 1:
        pages[0].save(base + ext, quality=92)
        return base + ext
    for i, pg in enumerate(pages, 1):
        pg.save(f"{base}_p{i:02d}{ext}", quality=92)
    return f"{base}_p01{ext}"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description="Turn clean rendered documents into realistic scans/photos, labels intact.")
    root = os.environ.get("DI_DATASET_ROOT") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    ap.add_argument("--out", default=root,
                    help="dataset root to read (default: dataset root; use ../irregular for the defective set)")
    ap.add_argument("--dest", default=None, help="where to write (default: <out>/degraded)")
    ap.add_argument("--levels", default="medium",
                    help="comma-separated: light,medium,heavy,photo,fax,mixed (one output per level)")
    ap.add_argument("--only", default="", help="comma-separated categories (default: all)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "jpg"],
                    help="pdf = image-only, no text layer, forces the OCR path (default)")
    ap.add_argument("--engine", default="augraphy", choices=["augraphy", "builtin"])
    ap.add_argument("--numba", action="store_true",
                    help="let augraphy use its numba kernels (faster, but its cache breaks on some installs)")
    ap.add_argument("--seed", type=int, default=42, help="same seed = same degradation")
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents per category")
    ap.add_argument("--force", action="store_true", help="redo documents that already exist")
    a = ap.parse_args()

    if a.numba:
        os.environ["NUMBA_DISABLE_JIT"] = "0"
    fitz = _pymupdf()
    _need("PIL", "pillow")

    levels = [x.strip() for x in a.levels.split(",") if x.strip()]
    bad = [x for x in levels if x not in PROFILES and x != "mixed"]
    if bad:
        raise SystemExit(f"unknown level(s) {bad}; choose from {sorted(PROFILES)} or 'mixed'")
    cats = CATS if not a.only else [c for c in CATS if c in {x.strip() for x in a.only.split(",")}]
    dest = a.dest or os.path.join(a.out, "degraded")
    engine = _augraphy if a.engine == "augraphy" else _builtin

    os.makedirs(os.path.join(dest, "labels"), exist_ok=True)
    made = skipped = missing = failed = 0
    for cat in cats:
        src_labels = os.path.join(a.out, "labels", cat + ".json")
        if not os.path.exists(src_labels):
            print(f"{cat}: no labels at {src_labels}, skipping"); continue
        records = json.load(open(src_labels, encoding="utf-8"))
        if a.limit:
            records = records[:a.limit]
        os.makedirs(os.path.join(dest, cat), exist_ok=True)
        out_records = []
        for rec in records:
            src_pdf = os.path.join(a.out, rec["file"])
            if not os.path.exists(src_pdf):
                missing += 1; continue
            stem = os.path.splitext(os.path.basename(rec["file"]))[0]
            for lv in levels:
                # seeded per (document, level) so adding documents never reshuffles the rest
                rng = random.Random(f"{a.seed}|{stem}|{lv}")
                prof = rng.choice(sorted(PROFILES)) if lv == "mixed" else lv
                tag = lv if lv != "mixed" else f"mixed-{prof}"
                base = os.path.join(dest, cat, f"{stem}__{tag}")
                probe = base + (".pdf" if a.format == "pdf" else "." + a.format)
                written, ops = probe, []
                if not a.force and os.path.exists(probe) and os.path.getsize(probe) > 0:
                    skipped += 1
                    ops = _ag_phases(prof, rng)[3] if a.engine == "augraphy" else []
                else:
                    try:
                        pages = _rasterize(fitz, src_pdf, PROFILES[prof]["dpi"])
                        imgs, ops = engine(pages, prof, rng)
                        written = _write(imgs, base, a.format, PROFILES[prof]["dpi"])
                        made += 1
                    except Exception as e:
                        print(f"FAILED {stem} [{tag}]: {type(e).__name__}: {e}")
                        failed += 1
                        continue
                out = dict(rec)
                out["file"] = os.path.relpath(written, dest).replace("\\", "/")
                out["source_file"] = rec["file"]
                out["degradation"] = dict(level=tag, profile=prof, engine=a.engine,
                                          dpi=PROFILES[prof]["dpi"], text_layer=False, ops=ops)
                out_records.append(out)
        json.dump(out_records, open(os.path.join(dest, "labels", cat + ".json"), "w",
                                    encoding="utf-8"), indent=2)
        print(f"{cat}: {len(out_records)} degraded records")

    print(f"\nengine={a.engine}  levels={','.join(levels)}  format={a.format}  seed={a.seed}")
    print(f"written={made} skipped={skipped}"
          + (f" failed={failed}" if failed else "")
          + (f" missing_source={missing}" if missing else ""))
    print(f"out={dest}")
    print("Labels keep their `irregularities`, so clean and defective sets both degrade unchanged.")


if __name__ == "__main__":
    main()
