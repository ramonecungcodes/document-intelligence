#!/usr/bin/env python3
"""Apply the routing policy to a run, and measure what it cost and bought.

    python -m route.cli apply --predictions reports/degraded-full.jsonl \\
        --corpus data/degraded
    python -m route.cli score --predictions reports/degraded-full.jsonl \\
        --corpus data/degraded

`apply` writes the review queue: which documents a person has to look at, and the gate
that sent each one. `score` grades the policy against the corpus -- what share was
accepted, how accurate the accepted half is, and how that compares to accepting the same
number of documents at random.

`score` is the one that matters, and it is why routing is a measured stage rather than a
configuration file. A policy that routes everything to a person is perfectly safe and
worth nothing; a policy that accepts everything is the pipeline with no routing at all.
Both look fine unless the accepted half is scored against a baseline, and the baseline
here is the same one used throughout Phase 5: sending the identical number of documents
to review at random, which leaves accuracy exactly where it started.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from core import config as config_mod
from route import policy as policy_mod

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "/data")
REPORTS_DIR = os.environ.get("DI_REPORTS_DIR", "/reports")


def rows_for(args):
    """Signals, outcome and a decision for every document in a run."""
    from eval.cli import signal_rows

    policy = policy_mod.build(config_mod.load(args.config))
    print(f"  {policy.describe()}")
    rows = signal_rows(args)
    for row in rows:
        row["decision"] = policy.decide(row["signals"])
    return policy, rows


def _rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def score(rows) -> dict:
    """What the policy accepted, and whether accepting it was better than luck."""
    accepted = [r for r in rows if not r["decision"].review]
    reviewed = [r for r in rows if r["decision"].review]
    baseline = _rate(sum(r["outcome"] for r in rows), len(rows))

    gates = {}
    for row in reviewed:
        for reason in row["decision"].reasons:
            block = gates.setdefault(reason.gate, {"fired": 0, "only_reason": 0,
                                                   "outcome": 0.0})
            block["fired"] += 1
            block["outcome"] += row["outcome"]
            if len(row["decision"].reasons) == 1:
                # What this gate caught that nothing else would have. A gate that never
                # fires alone is carried by its neighbours and could be switched off
                # without changing a single decision.
                block["only_reason"] += 1
    for block in gates.values():
        block["mean_outcome"] = round(block["outcome"] / block["fired"], 4)
        del block["outcome"]

    return {
        "documents": len(rows),
        "accepted": len(accepted),
        "reviewed": len(reviewed),
        "coverage": _rate(len(accepted), len(rows)),
        "accuracy_accepted": _rate(sum(r["outcome"] for r in accepted), len(accepted)),
        "accuracy_reviewed": _rate(sum(r["outcome"] for r in reviewed), len(reviewed)),
        "baseline": baseline,
        "lift": (round(_rate(sum(r["outcome"] for r in accepted), len(accepted))
                       - baseline, 4) if accepted and baseline is not None else None),
        # Documents that came back perfect and were sent to a person anyway. Not an
        # error -- a policy cannot know -- but the cost of the policy, and a number that
        # has to be visible or the coverage figure reads as free.
        "reviewed_but_perfect": sum(1 for r in reviewed if r["outcome"] >= 1.0),
        "accepted_but_wrong": sum(1 for r in accepted if r["outcome"] < 1.0),
        "gates": gates,
    }


def render(data: dict, policy) -> str:
    def pct(value):
        return "--" if value is None else f"{value * 100:.1f}%"

    out = ["", "ROUTING  -  who has to look at this", ""]
    out.append(f"  {policy.describe()}")
    out.append("")
    out.append(f"  documents            {data['documents']:>8}")
    out.append(f"  accepted             {data['accepted']:>8}   {pct(data['coverage'])}")
    out.append(f"  sent to review       {data['reviewed']:>8}")
    out.append("")
    out.append(f"  accuracy accepted    {pct(data['accuracy_accepted']):>8}")
    out.append(f"  accuracy reviewed    {pct(data['accuracy_reviewed']):>8}")
    out.append(f"  random at same rate  {pct(data['baseline']):>8}   "
               f"<- what routing at random would give")
    lift = data["lift"]
    out.append(f"  lift                 {'--' if lift is None else f'{lift:+.4f}':>8}")
    out.append("")
    out.append(f"  accepted but wrong   {data['accepted_but_wrong']:>8}   "
               f"escaped the gates")
    out.append(f"  reviewed but perfect {data['reviewed_but_perfect']:>8}   "
               f"the cost of the policy")

    if data["gates"]:
        out.append("")
        out.append(f"  {'gate':<26}{'fired':>7}{'alone':>7}{'mean outcome':>14}")
        for name, block in sorted(data["gates"].items(),
                                  key=lambda kv: -kv[1]["fired"]):
            out.append(f"  {name:<26}{block['fired']:>7}{block['only_reason']:>7}"
                       f"{block['mean_outcome']:>14.3f}")
        out.append("  `alone` is how often a gate was the only reason a document was")
        out.append("  routed. A gate that never fires alone changes no decisions.")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="route", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (("apply", "write the review queue"),
                            ("score", "grade the policy against the corpus")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--predictions", required=True)
        command.add_argument("--corpus", default=CORPUS_ROOT)
        command.add_argument("--config", default=None)
        command.add_argument("--only", default="")
        command.add_argument("--no-validators", action="store_true",
                             dest="no_validators")
        command.add_argument("--out", default=None)
        if name == "score":
            command.add_argument("--format", default="table",
                                 choices=["table", "json"])

    args = parser.parse_args(argv)
    policy, rows = rows_for(args)

    if args.command == "apply":
        queue = [{
            "file": row["file"],
            "doc_type": row["truth"],
            "profile": row["profile"],
            "why": row["decision"].to_dict()["why"],
            "reasons": [r.to_dict() for r in row["decision"].reasons],
            # Every signal, not only the ones that fired. A reviewer disagreeing with a
            # gate needs to see what the others said, and a queue that shows only the
            # triggering value cannot be argued with.
            "signals": row["signals"],
        } for row in rows if row["decision"].review]
        out = args.out or os.path.join(REPORTS_DIR, "review-queue.jsonl")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            for entry in queue:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  {len(queue)} of {len(rows)} documents need a person")
        print(f"  wrote {out}")
        return 0

    data = score(rows)
    if args.format == "json":
        sys.stdout.write(json.dumps(data, indent=1))
    else:
        print(render(data, policy))
    out = args.out or os.path.join(REPORTS_DIR, "routing.json")
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"policy": policy.describe(), **data}, handle, indent=1)
        if args.format != "json":
            print(f"routing written to {out}")
    except OSError as error:
        print(f"could not write {out}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
