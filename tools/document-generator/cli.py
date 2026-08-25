#!/usr/bin/env python3
"""Entrypoint for the di-document-generator container.

Wraps the three stages behind one command so callers do not need to know the
script names or the directory conventions.

    generate [flags]    synthetic HTML + ground-truth labels        -> $DI_DATASET_ROOT
    render   [flags]    HTML -> PDF via headless Chromium
    degrade  [flags]    PDF -> image-only scans/photos (no text layer)
    build    [flags]    the whole thing: clean set + defective set, rendered
                        (add --degrade to also write the scanned variants)

Anything after the subcommand is passed straight through, so every flag the
underlying scripts accept still works:

    build --multibill 60 --degrade --levels light,medium,photo
    generate --seed 99 --out /data/heldout
    degrade --out /data/irregular --levels fax
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DI_DATASET_ROOT") or os.path.abspath(os.path.join(HERE, ".."))
SCRIPTS = {"generate": "generate.py", "render": "render_pdfs.py", "degrade": "degrade.py"}


def run(stage, args):
    cmd = [sys.executable, os.path.join(HERE, SCRIPTS[stage]), *args]
    print(f"\n$ {stage} {' '.join(args)}", flush=True)
    code = subprocess.call(cmd)
    if code:
        raise SystemExit(f"{stage} failed (exit {code})")


def split_degrade_flags(args):
    """`build` owns --degrade; everything else is forwarded to generate/degrade."""
    want, passthrough, degrade_flags = False, [], []
    it = iter(args)
    for a in it:
        if a == "--degrade":
            want = True
        elif a in ("--levels", "--format", "--engine"):
            degrade_flags += [a, next(it, "")]
        else:
            passthrough.append(a)
    return want, passthrough, degrade_flags


def build(args):
    want_degrade, gen_flags, deg_flags = split_degrade_flags(args)
    irregular_root = os.path.join(ROOT, "irregular")

    run("generate", gen_flags)
    run("render", ["--out", ROOT])
    run("generate", [*gen_flags, "--irregular"])
    run("render", ["--out", irregular_root])

    if want_degrade:
        run("degrade", ["--out", ROOT, *deg_flags])
        run("degrade", ["--out", irregular_root, *deg_flags])

    print(f"\nDataset ready under {ROOT}")
    print("  clean set      ./            + ./labels")
    print("  defective set  ./irregular   + ./irregular/labels")
    if want_degrade:
        print("  scanned        ./degraded   and ./irregular/degraded")


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print(f"DI_DATASET_ROOT = {ROOT}")
        return 0
    stage, args = argv[0], argv[1:]
    if stage == "build":
        build(args)
    elif stage in SCRIPTS:
        run(stage, args)
    else:
        raise SystemExit(f"unknown command {stage!r}; try one of: "
                         f"{', '.join(['build', *SCRIPTS])}, help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
