# DocumentIntelligence

```
██████╗  ██████╗  ██████╗██╗   ██╗███╗   ███╗███████╗███╗   ██╗████████╗                    
██╔══██╗██╔═══██╗██╔════╝██║   ██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝                    
██║  ██║██║   ██║██║     ██║   ██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║                       
██║  ██║██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║                       
██████╔╝╚██████╔╝╚██████╗╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║   ██║                       
╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝                       
                                                                                            
██╗███╗   ██╗████████╗███████╗██╗     ██╗     ██╗ ██████╗ ███████╗███╗   ██╗ ██████╗███████╗
██║████╗  ██║╚══██╔══╝██╔════╝██║     ██║     ██║██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝
██║██╔██╗ ██║   ██║   █████╗  ██║     ██║     ██║██║  ███╗█████╗  ██╔██╗ ██║██║     █████╗  
██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██║     ██║██║   ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝  
██║██║ ╚████║   ██║   ███████╗███████╗███████╗██║╚██████╔╝███████╗██║ ╚████║╚██████╗███████╗
╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
```

An intelligent document processing system: point it at a folder, an inbox or a bucket,
and for each document it works out what the document is, extracts the fields that
matter, checks them against rules and against history, posts the ones it is confident
about, and asks a human a targeted question about the rest. Every answer a human gives
is written back, so the same question gets asked less often over time.

The interesting part is not that an LLM can read an invoice. It is being able to say
**how often it is right, and how you catch it when it is wrong.**

## Design

A **deterministic spine with agentic pockets.** Ordering, persistence, routing, retry
and export are ordinary code — replayable, diffable, auditable. The model's judgement
is confined to two places where a fixed sequence of calls genuinely cannot do the job:

- **Extraction on hard documents** — a tool-using loop that chooses which pages to
  read, when to zoom into an illegible region, when to check the vendor master before
  committing to a name.
- **The repair loop** — when validation fails, the extractor gets the document *and the
  specific failures* back and re-attempts, bounded.

Everything else stays fixed, because non-determinism at every stage would destroy the
accuracy measurement that is the entire point.

Nine stages, each a swappable plugin discovered through entry points:

```
source → normalize → split → classify → extract → validate → repair → decide → review/export
```

Two modes share almost all of that machinery: **Teach** (one document at a time, a
human confirming every field) and **Run** (batch, unattended, only the uncertain ones
surfaced). The review screen is the same in both — so anything a human corrects becomes
a labelled example whether they sat down to train the system or just cleared a queue.

What accumulates is a **knowledge pack**: document type definitions, per-sender layout
profiles, correction memory, learned validation rules, and threshold calibration. It is
a versioned directory you can read, diff and ship — not model weights. Which also means
the correction store *is* a fine-tuning dataset, generated as a by-product of normal use.

## Repo layout

```
core/                       shared domain code: field normalisation, type registry
extract/                    the extractor: PDF text -> schema-constrained fields
eval/                       scoring, the report format, the CLI
tests/                      unit tests, fixtured off the committed samples
tools/document-generator/   synthetic evaluation corpus (Docker; see its README)
data/                       generated corpus — gitignored, regenerable from a seed
reports/                    score reports — gitignored
docker-compose.yml          on-demand services: document-generator, evaluator
```

`core/normalize.py` holds the field comparison primitives — is `03/29/2026` the same
date as `2026-03-29`, is `Acme, Inc.` the same vendor as `Acme Inc`. They live in
`core` rather than `eval` because the validator plugins will ask exactly those
questions in production. One implementation, so evaluation and the pipeline can never
drift into disagreeing about whether a field is correct.

The remaining application packages (`plugins/`, `web/`, `worker/`, `packs/`) land here
as they are built, as further services off the same `di-app` image — they share a
codebase and a dependency set, so they share a build. The generator stays a separate
image because Chromium and the augraphy stack are ~2.4 GB the application never needs.

## Getting started

Only Docker is required.

```bash
docker compose run --rm --name di-document-generator document-generator build
```

That writes the full corpus to `./data`: clean documents, a parallel set with injected
defects, and ground-truth labels for both. Add the scanned variants — image-only PDFs
with no text layer, which force the OCR path and stop the accuracy numbers from
flattering themselves:

```bash
docker compose run --rm --name di-document-generator document-generator build --degrade --levels light,medium,heavy,photo,fax
```

Then extract and score:

```bash
export ANTHROPIC_API_KEY=...
docker compose run --rm extractor run --only invoices --limit 20
docker compose run --rm evaluator score --predictions /reports/predictions.jsonl
```

Both services sit behind the `tools` profile, so `docker compose up` starts nothing.
They are on-demand; `run --rm` gives one a job and throws the container away.

> **On Git Bash / MSYS (Windows):** an absolute container path passed as an argument is
> rewritten to a Windows path before Docker sees it, so `--corpus /data/degraded` fails
> looking for a directory under `C:/Program Files/Git`. Prefix with `MSYS_NO_PATHCONV=1`,
> or use PowerShell. Commands that take no path argument are unaffected.

## Measuring it

`eval/` grades predicted extractions against the corpus and writes a versioned JSON
report; the CLI is one renderer of it, and the `/eval` screen will be another.

Two baselines anchor the harness, and both are commands:

- **`selftest`** feeds the ground truth back in as predictions. It must score exactly
  `1.000`. If it ever drops below that, a normaliser is wrong and every number the
  project reports afterwards is quietly understated by a bug nobody would go looking
  for.
- **`score --predictions empty`** grades an extractor that ran and found nothing.

That second one is more interesting than it sounds. It does not score zero: the
defective corpus deliberately empties fields, and an extractor returning nothing
"agrees" about those. So the report carries accuracy **excluding blank fields**
alongside the raw number, and that is the honest one to quote — on the current corpus
the empty extractor scores `0.007` raw and `0.000` non-blank.

Everything is sliced by document type, layout and degradation profile rather than
reported as one number, because a blended average hides exactly what you need to know:
that you are fine on clean invoices and failing on faxed forms. Repeating groups —
invoice line items, the sections of a multi-bill invoice — get **row recall reported
separately from field accuracy**, since missing an entire billable service is a
different and worse failure than misreading one field inside a service you found.

Reports carry provenance and cost slots from version 1 even though nothing populates
them yet. A report that cannot say which corpus, model and knowledge pack produced it
is unattributable, and the calibration curve, the learning curve and the extractor
ablation are all comparisons across those axes.

## Extracting

`extract/` is the current extractor and it is deliberately the crudest thing that can
produce a number: read the PDF's text layer, send it once with a schema, keep what
comes back. No tools, no repair loop, no confidence, no retries on content. Every
later phase has to justify itself against whatever this scores.

The schema is generated from `core/doctypes.py` — the same declaration the scorer
grades against — so the extractor and the evaluator cannot drift into disagreeing
about what an invoice is. Every field is nullable and required, which lets the model
say "not on the document" without inventing a value or dropping the key. That matters
more than it sounds: a W-9 carries an SSN or an EIN and never both, and the defective
corpus empties fields on purpose.

Two things it deliberately does not do:

- **It does not classify.** The document type comes from the corpus. Phase 1 measures
  whether a model can read fields off a layout it has never seen; classification is a
  separate risk with its own phase, and mixing them would make a bad number impossible
  to attribute.
- **It does not correct the document.** The prompt tells the model to transcribe a
  total even when it disagrees with the line items. Detecting that disagreement is the
  validators' job, and an extractor that quietly fixes documents destroys the defect
  detection signal.

Scanned documents come back empty, because they have no text layer to read. That is
not a bug to paper over — it is the measured size of the gap OCR has to close, and the
reason normalisation becomes its own pipeline stage.

Runs report what they cost:

```
  extracted 20/20
  41,203 in / 8,914 out  ·  $0.43  ·  96s of model time
```

`python -m extract.cli schema --type multi_bill_invoice` prints the generated schema
and system prompt without calling anything.

## Why the corpus comes first

The build order is driven by **what risk each phase retires**, and every phase ends with
a number that says whether to continue. The corpus and the scoring harness come before
any pipeline code, so that the very first extractor has something to be measured against
— and so every change after it can be attributed.

The corpus covers five document types across 3–8 layouts each, a parallel defective set
with 40 tagged defect classes, and five degradation profiles from *clean office scanner*
to *photocopied then faxed*. Because every degraded document keeps its source label,
detection accuracy and extraction accuracy can be scored independently, per degradation
profile rather than as one blended number.

See [`tools/document-generator/README.md`](tools/document-generator/README.md) for the
label schemas, the defect catalogue and the degradation profiles. Six sample
documents are committed under
[`tools/document-generator/samples/`](tools/document-generator/samples/) — including a
multi-bill invoice, a defective one, and the same document degraded to an image-only
scan — so the output can be inspected without running anything.
