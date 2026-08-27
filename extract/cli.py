#!/usr/bin/env python3
"""Run the Phase 1 extractor over a corpus and write predictions.

    python -m extract.cli run --only invoices --limit 20
    python -m extract.cli run --model qwen/qwen3.5-9b --limit 5
    python -m extract.cli run --extractor anthropic --limit 5
    python -m extract.cli config
    python -m extract.cli schema --type multi_bill_invoice     # no API call

Predictions come out in the same shape as the corpus labels, so scoring them is:

    python -m eval.cli score --predictions /reports/predictions.jsonl

`schema` prints the generated JSON Schema and the system prompt for a type without
calling anything -- useful for seeing what the model is actually being asked for.

Which extractor runs, and how it is configured, both live in the manifest (di.toml):
choose a plugin and its settings are in the same block. `config` shows what resolved
and what every plugin accepts. `--extractor` and `--model` override for one run, which
is how the same corpus gets scored against two models.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
import re
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from core import config as config_mod
from core import doctypes
from core.plugins import describe
from eval.score import load_corpus
from extract import schema as schema_mod
from extract import backends
from extract.rules import RULES
from extract.backends import Usage
from extract.runner import extract_document

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "/data")
REPORTS_DIR = os.environ.get("DI_REPORTS_DIR", "/reports")


def sample(records, doctype, limit):
    """Take `limit` documents spread across the type's variants, not off the top.

    `records[:limit]` looked like a sample and was not. The first twelve forms in the
    corpus are all onboarding forms, so every quick check run during development graded
    onboarding and nothing else -- blind to claims, W-9s, W-4s and loan applications,
    three fifths of the type. A field that invented a co-applicant name on all 25 loan
    applications lacking one went unseen through a full day of regression runs, because
    a loan application was never once in the sample.

    Round-robin over the variants: first of each, then second of each. Any prefix is
    balanced, so `--limit 5` is as representative as `--limit 40`. Deterministic and
    order-preserving rather than random, because runs are compared against each other
    and a sample that moved between runs would make every comparison unreadable.

    Types without variants are stratified by layout instead, which is the other
    dimension the corpus varies deliberately.
    """
    if not limit or limit >= len(records):
        return records
    buckets = {}
    for record in records:
        key = (doctype.variant_of(record) if doctype.variant_key
               else str(record.get("layout", "")))
        buckets.setdefault(key, []).append(record)

    taken, depth = [], 0
    while len(taken) < limit:
        progressed = False
        for bucket in buckets.values():
            if depth >= len(bucket):
                continue
            taken.append(bucket[depth])
            progressed = True
            if len(taken) == limit:
                return taken
        if not progressed:      # every bucket exhausted before reaching the limit
            break
        depth += 1
    return taken


def from_manifest(path: str, limit: int):
    """Documents a splitter found, rather than documents the corpus declared.

    The manifest carries `truth_source` beside each piece so the run can be scored
    later. It is not read here and must not be: a piece whose type came from the
    corpus's answer key would make the splitter and the classifier look like they
    worked. Every piece goes through the classifier like any other unknown document,
    which is why this mode requires --type-from classifier.
    """
    with open(path, encoding="utf-8") as handle:
        pieces = json.load(handle)
    return [p["file"] for p in (pieces[:limit] if limit else pieces)]


def collect(corpus_root, only, limit):
    """Every corpus document, paired with the type declaration it should be read as."""
    jobs, unknown = [], set()
    for stem, records in load_corpus(corpus_root, only).items():
        doctype = doctypes.for_label_file(stem)
        if doctype is None:
            unknown.add(stem)
            continue
        for record in sample(records, doctype, limit):
            jobs.append((doctype, record["file"], doctype.variant_of(record)))
    return jobs, unknown


def predict_types(jobs, corpus_root, config, plugin, concurrency,
                  normalizer_name: str = ""):
    """Ask the classifier what each document is, instead of reading it off the corpus.

    This is the whole of Phase 3's claim. Every extraction number before it -- 0.986
    included -- was produced with the corpus handing over the answer, and a pipeline
    that cannot be run without its own ground truth has not been measured end to end.

    A document the classifier declines is not extracted. Grading an abstention as a
    document whose every field came back empty would score it zero and make declining
    look identical to failing, which is the reverse of the truth: the model said it
    could not read the page, and a page it cannot read is a page for a person. It is
    reported as coverage instead, alongside accuracy on what was attempted.
    """
    from classify.base import build as build_classifier
    from concurrent.futures import ThreadPoolExecutor

    # The runner's --normalizer has to reach a composite classifier too, or a cascade
    # will quietly build the manifest's engine while the extractor uses the one that
    # was asked for -- two different readings of the same page in one run.
    classifier = build_classifier(config=config, plugin=plugin,
                                  overrides={"normalizer": normalizer_name}
                                  if normalizer_name else None)
    normalizer = None
    if getattr(classifier, "NEEDS_TEXT", True):
        from normalize.base import NORMALIZERS, build as build_normalizer
        chosen = (config.chosen("normalizer") or "native").strip().lower()
        declares = {x.name for x in NORMALIZERS.get(
            chosen, type("x", (), {"SETTINGS": ()})).SETTINGS}
        normalizer = build_normalizer(
            config=config, overrides={"corpus": corpus_root} if "corpus" in declares else None)
    print(f"  classifier: {classifier.describe()}")

    def one(job):
        true_type, rel, true_variant = job
        path = os.path.join(corpus_root, rel)
        document = normalizer.read(path) if normalizer else None
        result = classifier.classify(document.text if document else "",
                                     document=document, path=path)
        return job, result

    predicted, abstained, wrong = [], [], []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for done, (job, result) in enumerate(pool.map(one, jobs), 1):
            true_type, rel, true_variant = job
            if result.abstained:
                abstained.append((rel, result.confidence))
                continue
            guess = doctypes.REGISTRY.get(result.doc_type)
            if guess is None:
                # A label the registry does not carry cannot select a schema. Treated
                # as an abstention rather than crashing the run.
                abstained.append((rel, result.confidence))
                continue
            if true_type is not None and (guess.name != true_type.name
                                          or result.variant != true_variant):
                wrong.append((rel, f"{true_type.name}/{true_variant}".rstrip("/"),
                              f"{guess.name}/{result.variant}".rstrip("/")))
            predicted.append((guess, rel, result.variant))
            if done % 100 == 0:
                print(f"  classified {done}/{len(jobs)}", flush=True)

    total = len(jobs)
    known = any(j[0] is not None for j in jobs)
    right = len(predicted) - len(wrong)
    if known:
        print(f"  type from classifier: {right}/{total} exact (type and variant), "
              f"{len(wrong)} wrong, {len(abstained)} declined")
    else:
        # Pieces from a splitter have no declared type to be right or wrong about.
        print(f"  type from classifier: {len(predicted)}/{total} typed, "
              f"{len(abstained)} declined")
    for rel, truth, guess in wrong[:10]:
        print(f"    {truth:26} read as {guess:26} {rel}")
    if len(wrong) > 10:
        print(f"    ... {len(wrong)} total")
    return predicted, abstained


def _validate_output(out_path: str, corpus_root: str, config_path: str) -> None:
    """Check what was just extracted, and say so beside it.

    Findings go in a sidecar rather than into the predictions. A prediction file is the
    extractor's answer and nothing else; mixing a second stage's opinion into it would
    make the two impossible to diff separately, and the scorer would be reading a file
    that two stages had written.

    A false alarm here means something specific, and only because the validators are
    gated on ground truth first: with rule correctness already established, a rule that
    fires on a document the corpus calls clean is the extractor being wrong. That makes
    this a defect report and an extraction-error detector at once.
    """
    from core import config as config_mod
    from validate.base import build_all, run as run_validators

    validators = build_all(config_mod.load(config_path))
    truth = {}
    for stem, records in load_corpus(corpus_root, None).items():
        doctype = doctypes.for_label_file(stem)
        if doctype is None:
            continue
        for record in records:
            truth[record["file"].replace("\\", "/")] = (doctype, record)

    findings, flagged, checked = [], 0, 0
    with open(out_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            predicted = json.loads(line)
            key = str(predicted.get("file", "")).replace("\\", "/")
            doctype, actual = truth.get(key, (None, None))
            if doctype is None:
                continue
            checked += 1
            report = run_validators(validators, predicted, doctype,
                                    doctype.variant_of(actual))
            if report.findings:
                flagged += 1
                findings.append({"file": key, **report.to_dict()})

    path = os.path.splitext(out_path)[0] + ".validation.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"validators": [v.name for v in validators],
                   "documents": checked, "flagged": flagged,
                   "findings": findings}, handle, indent=1)
    total = sum(len(f["findings"]) for f in findings)
    print(f"\nvalidators flagged {flagged}/{checked} documents ({total} findings)")
    print(f"      {path}")


def _cause(error: str) -> str:
    """The shape of a failure, with the document-specific detail stripped off.

    Twelve identical connection errors are one problem, not twelve, and printing them
    twelve times buries that. Grouping on the exception type plus the leading words of
    the message is enough to tell "the endpoint is unreachable" from "this page had bad
    JSON" without pretending to parse messages nobody controls.
    """
    head = error.split(":", 2)
    return ":".join(head[:2]).strip() if len(head) > 1 else error.strip()


def _resolve_out(value: str) -> str:
    """Resolve --out against the reports directory.

    A bare name is the intended usage. Absolute container paths are accepted but are
    a trap on Git Bash, where MSYS rewrites `/reports/x.jsonl` into a Windows path
    before Docker ever sees it -- the run then writes inside the container, and `--rm`
    deletes the results on exit. A rewritten path is detected and pulled back.
    """
    if not value:
        return os.path.join(REPORTS_DIR, "predictions.jsonl")
    mangled = re.match(r"^[A-Za-z]:[\/]", value)
    if mangled or not os.path.isabs(value):
        return os.path.join(REPORTS_DIR, os.path.basename(value))
    return value


def run(args):
    corpus_root = args.corpus
    if args.manifest:
        if args.type_from != "classifier":
            raise SystemExit(
                "--manifest holds documents a splitter found, so nothing knows their "
                "type yet.\n  Add --type-from classifier.")
        relatives = from_manifest(args.manifest, args.limit)
        # doctype is filled in by the classifier; None is a placeholder the
        # classification pass replaces, and nothing may read it before then.
        jobs, unknown = [(None, rel, "") for rel in relatives], set()
    else:
        jobs, unknown = collect(corpus_root, args.only, args.limit)
    for stem in sorted(unknown):
        print(f"skipping labels/{stem}.json: no document type registered", file=sys.stderr)
    if not jobs:
        raise SystemExit("nothing to extract")

    abstained = []
    if args.type_from == "classifier":
        config = config_mod.load(args.config)
        jobs, abstained = predict_types(jobs, corpus_root, config, args.classifier,
                                        args.concurrency, args.normalizer)
        if not jobs:
            raise SystemExit("the classifier declined every document")

    out_path = _resolve_out(args.out)
    if args.dry_run:
        print(f"{len(jobs)} documents (dry run, nothing is called)")
        for doctype, rel, variant in jobs[:10]:
            label = f"{doctype.name}/{variant}" if variant else doctype.name
            print(f"  {label:26} {rel}")
        print(f"  ... {len(jobs)} total")
        return 0

    overrides = {"model": args.model}
    if args.abort_after:
        # Cap the HTTP call too, so a stalled request cannot outlive the budget it is
        # being measured against.
        overrides["timeout"] = args.abort_after
    if args.no_think:
        overrides["no_think"] = True
    backend = backends.build(config=config_mod.load(args.config),
                             plugin=args.extractor, overrides=overrides)
    rules_fired = {}
    budget = f" · abort past {args.abort_after}s/doc" if args.abort_after else ""
    print(f"{len(jobs)} documents · {backend.describe()} · "
          f"{args.concurrency} at a time{budget}")
    total = Usage()
    done = failed = skipped = 0
    results = []

    config = config_mod.load(args.config)
    rule_settings = config.rules()
    try:
        active = [r.name for r in RULES.enabled(rule_settings)]
    except ValueError as error:
        raise SystemExit(f"configuration error: {error}")
    print(f"  rules: {', '.join(active) or 'none'}")

    # How text is obtained is a plugin choice like any other. `native` keeps Phase 1
    # behaviour; `cached` reads what a normalizer run already produced, so a degraded
    # corpus is extracted without this image carrying an OCR dependency.
    from normalize.base import NORMALIZERS, build as build_normalizer
    # A normalizer that keys a cache on corpus-relative paths has to agree with this
    # run about where the corpus starts. Making the caller set a second variable to
    # match --corpus is a trap: they disagree silently and every document reports as
    # uncached. This run already knows the answer, so it tells the plugin -- but only
    # if the plugin declares that setting, since binding an unknown key is an error
    # for good reason.
    chosen = (config.chosen("normalizer", args.normalizer) or "native").strip().lower()
    declares = {spec.name for spec in NORMALIZERS.get(chosen, type("x", (), {"SETTINGS": ()})).SETTINGS}
    overrides = {"corpus": corpus_root} if "corpus" in declares else None
    normalizer = build_normalizer(config=config, plugin=args.normalizer, overrides=overrides)
    print(f"  normalizer: {normalizer.describe()}")

    def work(job):
        doctype, rel, variant = job
        return extract_document(backend, doctype, os.path.join(corpus_root, rel), rel,
                                variant=variant, rule_settings=rule_settings,
                                normalizer=normalizer)

    # Stream each prediction to disk as it lands rather than accumulating in memory.
    # An hour of extraction should not be one crash away from nothing, and a partial
    # file is scoreable -- the scorer already reports what it could not match.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --resume keeps the successes of an interrupted run and redoes the rest. The
    # watchdog pattern depends on this: a run that stalls is killed and relaunched,
    # and without resume every relaunch starts from document one -- which turns one
    # hung request into a run that can never finish. Failures are retried on purpose;
    # a connection error at 2am should not be a permanent verdict.
    already = {}
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as previous:
            for line in previous:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue        # the line a kill interrupted mid-write
                if not record.get("_error"):
                    already[record["file"]] = record
    if already:
        jobs = [job for job in jobs if job[1] not in already]
        print(f"  resume: keeping {len(already)} finished, {len(jobs)} to go")

    stream = open(out_path, "w", encoding="utf-8", newline="\n")
    for record in already.values():
        stream.write(json.dumps(record) + "\n")
    stream.flush()

    aborted = ""
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {pool.submit(work, job) for job in jobs}
        try:
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    try:
                        result = future.result()
                    except Exception as error:
                        # A raise here used to escape the loop and silently abandon
                        # every document still pending -- 32 of 352 on the first full
                        # run, with the process still exiting 0.
                        done += 1
                        failed += 1
                        print(f"  [{done}/{len(jobs)}] FAILED (unhandled): "
                              f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
                        continue
                    done += 1
                    total.add(result.usage)
                    results.append(result)
                    stream.write(json.dumps(result.record) + "\n")
                    stream.flush()
                    if result.rules is not None:
                        for rule_name, n in result.rules.to_dict().items():
                            rules_fired[rule_name] = rules_fired.get(rule_name, 0) + n
                    name = result.record["file"]
                    if result.error:
                        failed += 1
                        print(f"  [{done}/{len(jobs)}] FAILED {name}: {result.error}",
                              file=sys.stderr, flush=True)
                    elif result.skipped:
                        skipped += 1
                    else:
                        print(f"  [{done}/{len(jobs)}] {name}  {result.usage.seconds:.0f}s",
                              flush=True)

                    if args.abort_after and result.usage.seconds > args.abort_after:
                        aborted = (f"{name} took {result.usage.seconds:.0f}s, over the "
                                   f"{args.abort_after}s budget")
                if aborted:
                    for future in pending:
                        future.cancel()
                    pending = set()
        finally:
            pass

    stream.close()

    if getattr(args, "validate", False):
        _validate_output(out_path, corpus_root, args.config)

    run_path = os.path.splitext(out_path)[0] + ".run.json"
    with open(run_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({
            "extractor": getattr(backend, "provenance", {"plugin": backend.name}),
            "corpus": corpus_root,
            "documents": len(jobs),
            "aborted": aborted or None,
            "rules": {"enabled": active, "changes": rules_fired},
            "failed": failed,
            "skipped_no_text_layer": skipped,
            # Documents the classifier declined. They are absent from the predictions
            # rather than present and empty: an abstention graded as a document whose
            # every field came back blank scores zero, which makes declining look
            # identical to failing. Coverage belongs next to accuracy, not inside it.
            "type_from": args.type_from,
            "declined_by_classifier": [
                {"file": rel, "confidence": conf} for rel, conf in abstained] or None,
            "usage": total.to_dict(),
        }, handle, indent=2)
        handle.write("\n")

    print()
    if aborted:
        print(f"ABORTED: {aborted}")
        print("  nothing is wrong with the run; the model is just too slow to iterate on.")
        print()
    if abstained:
        attempted = len(jobs)
        print(f"declined by the classifier: {len(abstained)} of "
              f"{attempted + len(abstained)} "
              f"({attempted / (attempted + len(abstained)):.1%} coverage)")
        print("  not extracted and not graded; they are the queue for a person.")
        print()
    print(f"wrote {out_path}")
    print(f"      {run_path}")
    if rules_fired:
        detail = ", ".join(f"{k} x{v}" for k, v in sorted(rules_fired.items()))
        print(f"  rules changed: {detail}")
    print(f"  extracted {done - failed - skipped}/{len(jobs)}"
          + (f", {failed} failed" if failed else "")
          + (f", {skipped} had no text layer" if skipped else ""))
    cost = f"${total.usd:.2f}" if total.usd else "$0.00 (local)"
    reasoning = f" ({total.reasoning_tokens:,} reasoning)" if total.reasoning_tokens else ""
    print(f"  {total.input_tokens:,} in / {total.output_tokens:,} out{reasoning}"
          f"  ·  {cost}  ·  {total.seconds:.0f}s of model time")
    print()

    # A run where nothing succeeded is a failed run, whatever the exit code says.
    #
    # Twice in one session a run wrote a file, printed a summary and exited 0 having
    # accomplished nothing: once when the Docker daemon was down, once when every
    # request failed to reach the model server. Both looked like results until the
    # predictions were read. Exiting non-zero and naming the shared cause turns a
    # silent no-op into something that cannot be mistaken for an answer.
    extracted = done - failed - skipped
    if not extracted and failed:
        causes = Counter(_cause(r.error) for r in results if r.error)
        cause, count = causes.most_common(1)[0]
        print(f"EVERY document failed. {count} of {failed} share one cause:")
        print(f"  {cause}")
        if len(causes) > 1:
            print(f"  ...and {len(causes) - 1} other kind(s); see the predictions file.")
        print()
        print("  Nothing was extracted, so there is nothing to score.")
        return 1

    print(f"score it:  python -m eval.cli score --predictions {out_path}")
    return 0


def show_config(args):
    """What the manifest resolves to, and what every plugin will accept."""
    config = config_mod.load(args.config)
    print(f"manifest: {config.path or '(none found; using defaults)'}")
    print()
    for slot in config_mod.SLOTS:
        chosen = config.chosen(slot)
        if chosen or slot == "extractor":
            print(f"  {slot:<12} {chosen or '(unset)'}")
    print()
    for name, backend_cls in sorted(backends.BACKENDS.items()):
        marker = "  <- selected" if name == (config.chosen("extractor") or "") else ""
        print(f"--- extractor: {name}{marker}")
        print(describe(name, backend_cls.SETTINGS))
        block = config.block("extractor", name)
        if block:
            try:
                resolved = config.settings("extractor", name, backend_cls.SETTINGS)
                from core.plugins import redact
                print("  resolved:", redact(resolved, backend_cls.SETTINGS))
            except Exception as error:
                print(f"  ERROR: {error}")
        print()
    if args.check:
        return _check_endpoint(config)
    return 0


def _check_endpoint(config) -> int:
    """Ask the endpoint whether it actually serves the model the manifest names.

    Nothing offline can catch this. The manifest pinned a model the server had never
    heard of and a base_url pointing at a workstation, and it went unnoticed for a
    fortnight because every run overrode both from the environment -- so it worked
    perfectly for the one person who already knew the right values, and failed on the
    first document for everyone else. That is the failure mode a committed manifest
    exists to prevent, so it is worth one round trip to confirm.
    """
    plugin = config.chosen("extractor") or "openai"
    try:
        backend = backends.build(config=config, plugin=plugin)
    except Exception as error:
        print(f"  cannot build {plugin}: {error}")
        return 1
    wanted = getattr(backend, "model", "")
    print(f"--- checking {plugin} against the endpoint")
    available = backend.available_models()
    if not available:
        print("  endpoint unreachable, or it lists no models. Nothing to check against.")
        return 1
    if wanted in available:
        print(f"  ok: {wanted} is served ({len(available)} models available)")
        return 0
    print(f"  MISSING: the manifest asks for {wanted!r}, which this endpoint does not serve.")
    close = [m for m in available if wanted.split("/")[-1][:6].lower() in m.lower()]
    for name in (close or available)[:8]:
        print(f"    available: {name}")
    return 1


def show_rules(args):
    """Every registered rule, which types it touches, and whether it is on."""
    config = config_mod.load(args.config)
    settings = config.rules()
    try:
        active = {r.name for r in RULES.enabled(settings)}
    except ValueError as error:
        raise SystemExit(f"configuration error: {error}")
    print(f"manifest: {config.path or '(none)'}")
    print()
    for rule in RULES:
        mark = "on " if rule.name in active else "off"
        scope = ", ".join(rule.applies_to) if rule.applies_to else "all types"
        print(f"  [{mark}] {rule.name:<22} {scope}")
        if rule.help:
            print(f"        {rule.help}")
    return 0


def show_schema(args):
    doctype = doctypes.REGISTRY.get(args.type)
    if doctype is None:
        raise SystemExit(f"unknown type {args.type!r}; known: {', '.join(sorted(doctypes.REGISTRY))}")
    print(schema_mod.instructions(doctype))
    print()
    print(json.dumps(schema_mod.json_schema(doctype), indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="extract", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    go = sub.add_parser("run", help="extract every document in the corpus")
    go.add_argument("--corpus", default=CORPUS_ROOT)
    go.add_argument("--only", default="", help="comma-separated label stems")
    go.add_argument("--limit", type=int, default=0,
                    help="N documents per type, spread across its variants")
    go.add_argument("--out", default=None, metavar="NAME",
                    help=f"file name inside {REPORTS_DIR} "
                         f"(default: predictions.jsonl)")
    go.add_argument("--normalizer", default="",
                    help="how text is obtained: native | cached | tesseract | ... "
                         "(see `normalize.cli engines`)")
    go.add_argument("--extractor", default="", help="which extractor plugin (see di.toml)")
    go.add_argument("--model", default="", help="override the chosen plugin's model")
    go.add_argument("--config", default="", help="path to the manifest (default: di.toml)")
    go.add_argument("--concurrency", type=int, default=4)
    go.add_argument("--abort-after", type=int, default=0, metavar="SECONDS",
                    help="stop the run if any single document takes longer than this")
    go.add_argument("--no-think", action="store_true",
                    help="disable chain-of-thought (overrides DI_NO_THINK)")
    go.add_argument("--resume", action="store_true",
                    help="keep successes already in --out; redo failures and the rest")
    go.add_argument("--dry-run", action="store_true", help="list the work, call nothing")
    go.add_argument("--type-from", default="corpus", choices=("corpus", "classifier"),
                    help="corpus: the label file says what each document is (the "
                         "measurement condition for phases 1 and 2). classifier: the "
                         "pipeline works it out, which is what production does.")
    go.add_argument("--classifier", default="",
                    help="which classifier plugin, when --type-from classifier")
    go.add_argument("--validate", action="store_true",
                    help="run the validators over what was extracted and write the "
                         "findings beside the predictions")
    go.add_argument("--manifest", default="",
                    help="extract the documents a splitter found (split.cli apply) "
                         "instead of the documents the corpus declares")

    show = sub.add_parser("schema", help="print the schema and prompt for a type")
    show.add_argument("--type", required=True)

    conf = sub.add_parser("config", help="show the resolved manifest and every setting")
    conf.add_argument("--config", default="")
    conf.add_argument("--check", action="store_true",
                      help="also ask the endpoint whether it serves the named model")

    rl = sub.add_parser("rules", help="show the post-extraction rules and their scope")
    rl.add_argument("--config", default="")

    args = parser.parse_args(argv)
    args.only = [s.strip() for s in args.only.split(",") if s.strip()] or None \
        if hasattr(args, "only") else None
    if args.command == "run":
        return run(args)
    if args.command == "config":
        return show_config(args)
    if args.command == "rules":
        return show_rules(args)
    return show_schema(args)


if __name__ == "__main__":
    sys.exit(main())
