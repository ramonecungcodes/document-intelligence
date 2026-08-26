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

## Phases

Ordered by risk rather than by feature, and each one ends with a number rather than a
demo. The question a phase answers is one that could kill the project, so the point of
finishing it is knowing whether to continue.

| phase | the risk it retires | ends with | state |
|---|---|---|---|
| **0** | Can any of this be measured? Corpus plus scoring harness, anchored by a self-test that must score exactly `1.000` and an empty-extractor baseline. | a scorer worth trusting | done |
| **1** | Can a model read fields off a document it has never seen? Text layer only, type given, one call, no tools, no repair. | a field-accuracy number | done — `0.973` |
| **2** | Can it read documents that are not clean? The normalizer becomes its own stage: degraded scans carry no text layer, so this is where OCR or a vision path has to earn its place. | degraded accuracy against clean | next |
| **3** | Can it tell what a document *is*? Type moves from corpus-given to predicted, and the splitter handles files holding more than one document. | classification accuracy | |
| **4** | Can it tell when it is wrong? Validators: arithmetic that must foot, dates that must parse, cross-field constraints that must hold. | defect precision and recall | |
| **5** | Is its confidence real? Calibration from independent signals rather than model self-report, and routing what fails to a person. | a calibration curve | |
| **6** | Can it repair itself? The bounded repair loop and tool-using extraction — the agentic pockets, arriving last because everything before them is what makes them measurable. | repair success rate | |
| **7** | Does teaching it work? Teach mode and the run queue as one screen, with the knowledge pack accumulating corrections, layout profiles and learned validators. | a learning curve | |

Two consequences of this order are visible in the code and worth naming.

Phase 1 deliberately refuses to classify. The document type comes from the corpus,
because mixing extraction and classification would make a bad number impossible to
attribute to either. The same reasoning keeps OCR out until Phase 2: documents with no
text layer come back empty and are reported as skipped rather than scored, so the size
of the gap OCR has to close is a measurement instead of an assumption.

And the phases are not independent. Everything after Phase 1 measures itself against
extraction output, which is why so much of Phase 1 went into the harness rather than
the model. A validator cannot tell you whether the rule or the extractor is wrong. A
confidence score cannot be calibrated against a systematically biased signal. Worst of
all, a learning loop built on a broken extraction contract still appears to learn: the
knowledge pack fills with thousands of corrections that all encode one schema defect,
accuracy climbs, and the system is being taught to compensate for a bug rather than to
read documents.

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

## Results

Phase 1, 352 documents, `qwen3-vl-8b` running locally through an OpenAI-compatible
endpoint. Both columns are the same scorer over both prediction files, because the
scorer changed during the phase and comparing across versions would be meaningless.

| slice | first baseline | current |
|---|---|---|
| **overall field accuracy** | 0.928 | **0.973** |
| purchase orders | 0.966 | **1.000** |
| multi-bill invoices | 0.567 | **0.977** |
| invoices | 0.976 | 0.979 |
| forms | 0.977 | 0.975 |
| resumes | 0.803 | 0.832 |
| documents scored / failed | 329 / 8 | 340 / 12 |

Almost all of the gain is multi-bill invoices, and none of it came from a better model.
Seven consecutive multi-bill defects were traced, and every one was in the harness: a
field described identically to two siblings so the model rotated them, a field the
generator never printed, a label that was a prefix of its own value, a description
whose generic half contradicted its specific half. The model was reading the documents
correctly the whole time and being asked the wrong questions.

The remaining 12 failures are a single cause -- long resumes exceeding the token budget
mid-answer -- which the runner used to report as malformed JSON.

### Held out

Every fix above was developed against the same 12 multi-bill documents, which is how
you fit noise. The other 28 in the corpus were never extracted until the phase ended:

| field | fitted (12 docs) | held out (28 docs) |
|---|---|---|
| `reference_number` | 1.000 | 1.000 |
| `cost_center` | 0.778 | 0.800 |
| `service_location` | 0.556 | 0.600 |
| line items (4 fields, 184 rows) | 1.000 | 1.000 / 0.995 |

No degradation. The two weak fields stayed weak at the same rate, which is the right
kind of boring: it says the gap is a property of the task rather than of the sample.

### The extractor ablation

Same 12 documents, same prompts, three models:

| model | completed | output tokens | wall clock | notes |
|---|---|---|---|---|
| `qwen3-vl-8b` | 12/12 | 10,769 | 684s | the baseline |
| `qwen3-vl-30b` | 12/12 | 10,309 | 581s | **worse**, and in the same places |
| `deepseek-r1-8b` | 3/12 | 82,210 | 4,393s | 95.7% of output was reasoning |

Four times the parameters made it worse, not better -- more truncation errors, plus two
arithmetic mistakes the smaller model did not make. The reasoning model spent nearly
all of its budget thinking and ran out of tokens mid-answer on three quarters of the
documents. Reading a value printed on a page does not benefit from deliberation.

One caveat stated rather than buried: the reasoning model ran unconstrained, because
that endpoint rejects the schema, so its accuracy is not a like-for-like comparison.
Its token cost and completion rate are unaffected by that and stand on their own.

### Knowing what it does not know

A single accuracy figure charges the same penalty for a value that was missed and a
value that was invented. Those are not equally bad -- a blank field is honest and gets
looked at, a confident wrong one flows downstream unchallenged -- so fields the
document may legitimately omit are also scored on whether the extractor noticed:

| field | absent cases | invented | copied from a neighbour |
|---|---|---|---|
| `co_applicant_name` | 25 | 1.000 | 0.000 |
| `service_location` | 46 | 1.000 | 0.804 |
| `business_name` | 16 | 0.938 | 0.000 |
| `ssn` / `ein` | 20 | **0.000** | 0.000 |

Two different failures. `service_location` fills an empty slot from the field next to
it; `co_applicant_name` invents a person outright on every loan application that has
none. Both were invisible in an accuracy column, and both were being counted correctly
the whole time -- the scorer recorded them from the first version and the renderer
never printed them.

`ssn` and `ein` are the reason this is a design problem rather than a model limit. They
are mutually exclusive on a W-9, and the extractor abstains on the absent one every
time. The same model, on the same run, knows how to return nothing.

### Asking a different question

Three rounds of rewriting told the model, in escalating detail, that absence was normal
and that borrowing a cost centre was wrong. The last of them named the exact wrong
answers. Those rewrites moved the count 16, 16, 16. A model four times larger also
produced 16.

So the ask changed shape instead. A field the document may legitimately omit is no
longer a nullable slot; it is a decision the model has to return:

```json
"service_location": { "status": "present | absent | unclear", "value": null }
```

Same model, same documents, same prompt text:

| field | invented, before | invented, after |
|---|---|---|
| `co_applicant_name` | 1.000 | **0.000** |
| `business_name` | 0.938 | **0.000** |
| `service_location` | 1.000 | **0.312** |
| `ssn` / `ein` (control) | 0.000 | 0.000 |

Outright fabrication stopped completely. `co_applicant_name` has no field description
at all -- it was left undocumented on purpose, so the typed decision was the only thing
that changed. `service_location` improved but is not fixed: where the borrowed value
sits physically beside the field on the page, one in three still gets taken.

The runner collapses the decision back to a plain value, so rules, scorer and stored
records never see the shape. `unclear` is discarded rather than kept, because this
field is worth optimising for precision -- an invented address flows downstream
unchallenged while a blank one gets looked at -- and it stays a distinct answer because
it is what confidence routing consumes later.

The general lesson is the one the whole phase kept teaching. Every defect found here
was in the harness, not the model: a self-contradicting description, a field the
generator never printed, a key the schema asked the model to supply after using it to
choose that schema, a counter recorded on every run and never printed. Nullability was
never the missing piece. The type already permitted null and the prose already begged
for it. What the model could not do was return an answer the schema had no way to ask
for.

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
