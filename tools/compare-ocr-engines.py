"""Rank three OCR engines over one cohort, paired per document.

The aggregate score table cannot answer "is v6 better than docTR" -- it reports each
engine's accuracy separately, and the difference between two independent-looking numbers
carries no interval. The engines were run over the IDENTICAL 75 documents with the
identical model and prompts, so the comparison is paired: each document's difference is
observed directly, and the variance of the difference is far smaller than the variance
of either engine.

Clusters are source documents (claim_1__fax and claim_1__light are two photographs of
one page), because three degradations of one page do not move independently.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import score as score_mod
from eval.repair import cluster_bootstrap

CORPUS = "data/degraded"
ENGINES = [("docTR", "reports/ocr-doctr75.jsonl"),
           ("PP-OCRv5", "reports/ocr-paddle75.jsonl"),
           ("PP-OCRv6", "reports/ocr-paddlev6-75.jsonl")]


def profile(name):
    stem = name.rsplit("/", 1)[-1]
    return stem.split("__")[-1].replace(".pdf", "") if "__" in stem else "clean"


def cluster(name):
    return name.split("__")[0]


per = {}
for label, path in ENGINES:
    per[label] = score_mod.per_document(CORPUS, score_mod.load_records(path))
    print("%-9s scored %d documents" % (label, len(per[label])))

# The cohort must be identical or the comparison is between different questions.
sets = [set(v) for v in per.values()]
common = set.intersection(*sets)
for label, table in per.items():
    extra = set(table) - common
    if extra:
        print("  WARNING %s scored %d documents the others did not" % (label, len(extra)))
print("common cohort: %d documents\n" % len(common))

print("ACCURACY BY PROFILE  (field-weighted, over the common cohort)")
print("  %-9s %8s %8s %8s %8s" % ("engine", "all", "fax", "light", "photo"))
for label, table in per.items():
    row = ["%-9s" % label]
    for prof in (None, "fax", "light", "photo"):
        num = den = 0
        for key in common:
            if prof and profile(key) != prof:
                continue
            rec = table[key]
            if rec["failed"] or not rec["fields_graded"]:
                continue
            num += rec["fields_correct"]
            den += rec["fields_graded"]
        row.append("%8.4f" % (num / den) if den else "%8s" % "--")
    print("  " + " ".join(row))

print("\nPAIRED DIFFERENCES  (per document, clustered on the source page)")
pairs = [("PP-OCRv6", "docTR"), ("PP-OCRv6", "PP-OCRv5"), ("PP-OCRv5", "docTR")]
for prof in (None, "fax", "light", "photo"):
    print("\n  %s" % (prof or "all profiles"))
    for a, b in pairs:
        by_cluster, better, worse, same = {}, 0, 0, 0
        for key in sorted(common):
            if prof and profile(key) != prof:
                continue
            one, two = per[a][key], per[b][key]
            if one["failed"] or two["failed"]:
                continue
            if not one["fields_graded"] or not two["fields_graded"]:
                continue
            # Field accuracy per document, so a 24-field form does not outvote a
            # 9-field one purely by length.
            delta = (one["fields_correct"] / one["fields_graded"]
                     - two["fields_correct"] / two["fields_graded"])
            by_cluster.setdefault(cluster(key), []).append(delta)
            if delta > 1e-9:
                better += 1
            elif delta < -1e-9:
                worse += 1
            else:
                same += 1
        flat = [d for vals in by_cluster.values() for d in vals]
        if not flat:
            continue
        mean = sum(flat) / len(flat)
        interval = cluster_bootstrap(by_cluster)
        if interval is None:
            verdict = "too few clusters to resample"
        else:
            low, high = interval
            resolvable = (low > 0) or (high < 0)
            verdict = "[%+.4f, %+.4f] %s" % (
                low, high, "resolvable" if resolvable else "spans zero")
        print("    %-9s vs %-9s  %+.4f  %-34s  %d better / %d worse / %d tied"
              % (a, b, mean, verdict, better, worse, same))
