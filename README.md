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
| **1** | Can a model read fields off a document it has never seen? Text layer only, type given, one call, no tools, no repair. | a field-accuracy number | done |
| **2** | Can it read documents that are not clean? The normalizer becomes its own stage: degraded scans carry no text layer, so this is where OCR or a vision path has to earn its place. | degraded accuracy against clean | done |
| **3** | Can it tell what a document *is*? Type moves from corpus-given to predicted, and the splitter handles files holding more than one document. | classification accuracy | classifier done; splitter blocked |
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
normalize/                  the OCR stage: native text, tesseract, docTR, cascade
classify/                   the classifier stage: keyword baseline, llm, layout, dit
extract/                    the extractor: document text -> schema-constrained fields
eval/                       scoring, the report format, the CLI
tests/                      unit tests, fixtured off the committed samples
tools/document-generator/   synthetic evaluation corpus (Docker; see its README)
data/                       generated corpus — gitignored, regenerable from a seed
reports/                    score reports — gitignored
docker-compose.yml          on-demand services: generator, normalizer, extractor, evaluator
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
alongside the raw number, and that is the honest one to quote. Run it and see the gap
for yourself: the raw figure sits meaningfully above zero, and that difference is the
free credit any extractor would otherwise be quietly collecting.

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

Numbers live in the write-up rather than here, deliberately. A README that quotes a
score acquires a maintenance burden it will lose -- the figure goes stale the next time
anything changes, and a stale number in the first file a reader opens is worse than no
number at all.

Reproduce them instead. The corpus is regenerable from its seed and every run writes a
versioned report carrying the model, the manifest and the resolved settings that
produced it:

```bash
docker compose run --rm document-generator build          # corpus, from seed 42
docker compose run --rm extractor config --check          # confirm the endpoint serves the model
docker compose run --rm extractor run --out mine.jsonl    # extract
docker compose run --rm evaluator score --predictions /reports/mine.jsonl
```

`score --predictions self` must return exactly `1.000`, and `--predictions empty` gives
the do-nothing baseline. Anything you measure sits between those two.

## Phase 2: reading documents that are not clean

Phase 1 read the embedded text layer and nothing else, which is fine until a document
arrives as a photograph of a crumpled page. The generator produces 1,056 degraded
variants of the corpus at three levels -- a decent office scanner, a 170 dpi fax, and a
phone snapshot with perspective distortion -- as image-only PDFs with no text layer at
all.

The honest starting number was not a low score. It was **no score**: the extractor read
zero characters from every one of the 1,056 and reported them skipped. That is the gap
OCR had to close, measured rather than assumed.

### Which OCR engine, measured rather than argued

The normalizer is a plugin slot with competing implementations, so the comparison is a
manifest edit. Seventy-five degraded documents, stratified across every type and
profile, same model and prompts throughout -- only the OCR engine varies:

| profile | n | cascade | docTR | delta |
|---|---|---|---|---|
| light | 29 | `0.905` | **`0.921`** | +1.6 |
| photo | 18 | `0.502` | **`0.817`** | **+31.5** |
| fax | 28 | `0.292` | **`0.305`** | +1.3 |

Against `0.986` on the same documents before they were degraded.

**Light degradation is close to solved.** An office-scanner document costs about six
points end to end, which is shippable.

**Photographs are an OCR problem, not a model problem.** The same model on the same
pages scores `0.502` or `0.817` depending purely on which engine produced the text.
Nothing about the prompt or the schema moves that number; the perspective distortion
and uneven lighting are recovered by docTR's detector and lost by Tesseract's.

**Faxes are not recoverable.** `0.305`, and no prompting fixes characters OCR never
produced. The right system response is confidence routing to a person, and knowing
where that threshold sits is what this phase bought.

### The optimisation that cost 31 points

The cascade -- cheap engine first, escalate to the expensive one only where confidence
is poor -- is a good pattern and was the wrong choice here.

Its entire argument was cost, and the cost was not real. It averaged 3.6s a document
against docTR's 4.0s, because on faxes it runs *both* engines. Seven minutes saved
across the corpus, on a stage that is cached and paid once, against 7.4 hours of
extraction. An optimisation worth 1.4% of the pipeline.

Meanwhile it lost 31 points on photographs, and the reason is worth keeping. Tesseract
reports around `0.88` mean word confidence on photo documents whose text then extracts
at `0.191`. Its confidence is well calibrated on clean scans and badly calibrated on
distorted ones -- so the escalation rule, which trusted that number, kept exactly the
documents it should have escalated.

**A confidence signal is only useful where it is calibrated, and the place you most
want to trust it is the place least likely to be true.** That is the lesson, and it
lands squarely on Phase 5.

The cascade stays in the codebase as a plugin. It remains the right pattern where OCR
is not cached -- per-request production traffic, CPU-only deployment, an engine billed
per page -- and running both engines over the same documents is what produced the
comparison above. It is simply not what this pipeline should default to.

## Phase 3: working out what a document is

Phases 1 and 2 were handed the document type by the corpus. That was deliberate —
mixing extraction and classification would have made a bad number impossible to
attribute to either — but it is not how the system ever runs. A folder of scanned
paperwork does not come labelled.

Five types, so the floor is not zero. Always answering the commonest type scores
`0.200` on a balanced sample having read nothing, and every result below is reported
against that.

| classifier | reads | clean | fax |
|---|---|---|---|
| keyword | printed phrases | `0.700` | — |
| llm | OCR text | `0.990` | `0.571` |
| LayoutLMv3 | words + boxes + image | `1.000` | `0.943` |
| **dit** | **the page image alone** | **`1.000`** | **`0.958`** |

The keyword baseline is a floor, not a candidate: its phrases are drawn from what the
generator prints, which real vendors would not oblige it by repeating. It reaches
`0.350` on multi-bill invoices, which say "Invoice" exactly as loudly as invoices do.

### The fax gap is not comprehension

The LLM loses 42 points between clean documents and faxes, and the cause is upstream of
the model. docTR finds **62% of the words** on a 170 dpi bitonal page — the rest are not
misread, they are never found. No amount of reading recovers a word that was never
there.

Geometry survives what glyphs do not, and that was checked before anything was trained.
A coarse ink-occupancy grid — no words in it at all, just where marks sit on the page —
nearest-neighboured against the clean corpus:

| profile | word retention | layout fidelity | type accuracy from ink alone |
|---|---|---|---|
| light | `0.954` | `0.983` | `1.000` |
| photo | `0.938` | `0.958` | `0.997` |
| fax | `0.645` | `0.772` | `0.875` |

A fax keeps two-thirds of its words and three-quarters of its layout, and position
alone identifies the type better than the LLM reading the text does.

### Dropping the text made it better

The result worth keeping is that **the image-only model beats the multimodal one**.
LayoutLMv3 gets the words, the word boxes *and* the page; DiT gets only the page; DiT
wins on every profile and trains in a fifth of the time.

That is not an argument against fusion in general — LayoutLMv3 *is* fusion, learned and
early, which is strictly more expressive than averaging two models' probabilities. It is
an argument about *these* documents. The failure modes are correlated: DiT's unsure
documents carry a mean of 43 OCR words against 92.5 for its confident ones, so a text
branch goes blind exactly where the image branch needs help. Fusion pays when errors are
independent, and degradation takes out both at once.

### Held out by page design, not by document

A perfect score on a generated corpus is a reason for suspicion. Every document of a
type is drawn from a handful of designs, so holding out *documents* cannot separate a
model that learned what an invoice is from one that memorised what this corpus's
invoices look like. Measured directly: DiT scores `1.000` on faxes held out by document
and `0.958` held out by page design. The gap is what it had memorised.

The numbers above are the design-holdout ones. Every document exists four times — clean,
light, photo, fax — so the split is by source document too; training on a document's
light version while testing its fax version would measure memorisation of that document.

### Confidence does the routing

`0.958` is not the number the pipeline runs on. Every error DiT made on unseen-design
faxes arrived below `0.90` confidence, so that is where `di.toml` sets the floor:

| profile | coverage at 0.90 | accuracy when it answers |
|---|---|---|
| clean | 100% | `1.000` |
| light | 100% | `1.000` |
| photo | 98.6% | `1.000` |
| fax | 88.7% | `1.000` |

One threshold, no errors anywhere it commits, and it costs essentially nothing on the
profiles that were already solved. The documents it declines are the ones there was
least to read on — the 43-against-92.5 word gap above is the same split. It abstains
where the page has gone blank, which is the typed-decision discipline Phase 1 applied
to fields, arriving one stage earlier.

And because DiT reads no text, this stage runs *before* the normalizer: classification
costs a page render rather than an OCR pass over the corpus.

### What this phase did not close

The **splitter has no corpus**. Zero multi-document files exist, so the risk that a
scanned batch holds three documents in one PDF is untested — it needs generator work
before it can be measured, and claiming it works on the strength of the classifier
would be exactly the unattributable number this project exists to avoid.

Extraction still receives the *true* type. Wiring the predicted type through and
measuring what classification costs downstream is the remaining piece.

These figures also predate the ten-design generator: invoices and purchase orders
carried three and two designs when they were measured, and the design holdout could not
be applied to those two types at all. Regenerating is what turns that from an untested
assumption into a number.

## Why the corpus comes first

The build order is driven by **what risk each phase retires**, and every phase ends with
a number that says whether to continue. The corpus and the scoring harness come before
any pipeline code, so that the very first extractor has something to be measured against
— and so every change after it can be attributed.

The corpus covers five document types — ten page designs each for invoices, purchase
orders and multi-bill invoices, three for forms and eight for resumes — a parallel defective set
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
