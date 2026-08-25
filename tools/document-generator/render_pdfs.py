#!/usr/bin/env python3
"""Render the generated HTML documents to PDF using headless Chrome.

Usage:
  python render_pdfs.py                       # renders ../source_html/* -> ../<category>/*.pdf
  python render_pdfs.py --out ./heldout       # render a set produced with a custom --out
  python render_pdfs.py --degrade             # then also write ../degraded/ scans
  python render_pdfs.py --degrade --degrade-levels light,medium,heavy,photo,fax

Chrome renders a perfect text layer, so --degrade hands the finished PDFs to
degrade.py, which writes image-only scans that force the OCR path instead.

Requires Google Chrome (or Edge). Adjust CHROME below if installed elsewhere.
"""
import os, sys, subprocess, glob, argparse, tempfile

CANDIDATES = [
    os.environ.get("CHROME_BIN", ""),                       # set by the Docker image
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable",
]
CHROME = next((c for c in CANDIDATES if c and os.path.exists(c)), None)

# Chromium cannot use its sandbox as root inside a container, and /dev/shm is small
# there. The image sets these; a normal desktop run leaves them off.
EXTRA = ["--no-sandbox", "--disable-dev-shm-usage"] if os.environ.get("CHROME_NO_SANDBOX") else []

DATASET_ROOT = os.environ.get("DI_DATASET_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DATASET_ROOT)
    ap.add_argument("--only", default="", help="comma-separated categories to render (default: all)")
    ap.add_argument("--force", action="store_true", help="re-render even if the PDF is up to date")
    ap.add_argument("--degrade", action="store_true",
                    help="after rendering, write image-only scans to <out>/degraded (see degrade.py)")
    ap.add_argument("--degrade-levels", default="medium",
                    help="profiles to degrade with: light,medium,heavy,photo,fax,mixed")
    a = ap.parse_args()
    # Chrome's --print-to-pdf silently writes nothing for a relative path, so a
    # documented invocation like `--out ../irregular` would produce zero PDFs.
    a.out = os.path.abspath(a.out)
    if not CHROME:
        raise SystemExit("No Chrome/Edge found; edit CANDIDATES in render_pdfs.py")
    cats = ["invoices", "purchase_orders", "resumes", "forms", "multi_bill_invoices"]
    if a.only:
        want = {c.strip() for c in a.only.split(",")}
        cats = [c for c in cats if c in want]
    profile = tempfile.mkdtemp(prefix="chrome-pdf-")
    total = ok = skipped = 0
    for cat in cats:
        src = os.path.join(a.out, "source_html", cat)
        dst = os.path.join(a.out, cat); os.makedirs(dst, exist_ok=True)
        files = sorted(glob.glob(os.path.join(src, "*.html")))
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            outpdf = os.path.join(dst, name + ".pdf")
            if (not a.force and os.path.exists(outpdf) and os.path.getsize(outpdf) > 0
                    and os.path.getmtime(outpdf) >= os.path.getmtime(f)):
                ok += 1; skipped += 1; continue
            url = "file:///" + os.path.abspath(f).replace("\\", "/")
            cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
                   "--no-pdf-header-footer", f"--user-data-dir={profile}",
                   *EXTRA, f"--print-to-pdf={outpdf}", url]
            try:
                subprocess.run(cmd, capture_output=True, timeout=90)
            except Exception as e:
                print("ERROR", name, e)
            total += 1
            if os.path.exists(outpdf) and os.path.getsize(outpdf) > 0:
                ok += 1
            else:
                print("FAILED", name)
        print(f"{cat}: {len(files)} html")
    print(f"done: {ok}/{total} PDFs present in {a.out} (skipped {skipped} already up to date)")

    if a.degrade:
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "degrade.py"),
               "--out", a.out, "--levels", a.degrade_levels]
        if a.only:
            cmd += ["--only", a.only]
        if a.force:
            cmd += ["--force"]
        print()
        print("degrading -> " + os.path.join(a.out, "degraded"))
        raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
