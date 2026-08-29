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
| **3** | Can it tell what a document *is*? Type moves from corpus-given to predicted, and the splitter handles files holding more than one document. | classification accuracy | done |
| **4** | Can it tell when it is wrong? Validators: arithmetic that must foot, dates that must parse, cross-field constraints that must hold. | defect precision and recall | done |
| **5** | Is its confidence real? Calibration from independent signals rather than model self-report, and routing what fails to a person. | a calibration curve | done |
| **6** | Can it repair itself? The bounded repair loop and tool-using extraction — the agentic pockets, arriving last because everything before them is what makes them measurable. | repair success rate | repair loop done; tool use deferred |
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
classify/                   the classifier stage: keyword baseline, llm, layout, dit, cascade
split/                      the splitter stage: single, every_page, by_type
validate/                   the validator stage: arithmetic, required, format, range, temporal
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

### Revisited after Phase 6: the engine choice was worth more than the loop

The table above ranked two engines on the cohort as it stood then. Phase 6 built the
machinery to compare two systems *properly* -- paired per document, resampled over
source pages -- and it was worth pointing that machinery back at this decision, with a
third engine added.

Three engines, the identical 75 documents, identical model and prompts. Only the engine
varies. `tools/compare-ocr-engines.py` reproduces it.

| engine | all | fax | light | photo |
|---|---|---|---|---|
| docTR (incumbent) | `0.657` | `0.285` | `0.902` | `0.826` |
| PaddleOCR PP-OCRv5 | `0.653` | `0.248` | `0.892` | `0.878` |
| **PaddleOCR PP-OCRv6** | **`0.702`** | **`0.336`** | **`0.925`** | **`0.897`** |

Those are three separate numbers per row and the difference between two of them carries
no interval, which is exactly the mistake this project keeps finding in its own work. The
comparison that means something is paired, because every engine read the same page:

| comparison | delta | interval | |
|---|---|---|---|
| PP-OCRv6 vs docTR, all | **`+0.052`** | [+0.022, +0.088] | resolvable |
| PP-OCRv6 vs docTR, fax | **`+0.058`** | [+0.018, +0.099] | resolvable |
| PP-OCRv6 vs docTR, photo | **`+0.077`** | [+0.005, +0.152] | resolvable |
| PP-OCRv6 vs docTR, light | `+0.031` | [-0.008, +0.077] | spans zero |
| PP-OCRv5 vs docTR, all | `-0.000` | [-0.037, +0.036] | spans zero |

**PP-OCRv5 was not worth switching to.** Dead level with docTR at `-0.0001` -- better on
photographs, worse on faxes, neither resolvable. It looked promising on an unpaired
glance at aggregate numbers, which is the comparison that carries no interval.

**PP-OCRv6 is, and it wins where the accuracy actually is.** Resolvably better on fax and
photo, the two profiles that were breaking the pipeline, and not resolvable on `light`
where all three engines are already above `0.89` and there is nothing left to win. A gain
concentrated on the hard profiles and absent on the easy one is the right shape for an
engine difference; the reverse would suggest measurement noise.

One caveat belongs beside the number. There is no same-engine control arm here, so this
delta is *the engine plus one sample of extractor noise*. Phase 6 sized that noise
directly -- a blind re-run of the identical request on degraded documents moved accuracy
by `-0.010` -- and the gain here is five times that, on 28 documents better against 9
worse. It survives the caveat. It is not free of it.

**The comparison against Phase 6 is the point.** Perfect selection of which documents to
repair -- the most interesting question that phase left open -- is worth `+0.003`.
Changing one line in the manifest is worth `+0.052`. The loop was the harder engineering
and the smaller number, and only a paired comparison makes those two commensurable enough
to say so.

The engine has not been switched here. That is a corpus-wide re-normalisation and every
downstream number in this README was measured through docTR; changing it silently would
invalidate the lot. It is Phase 7's first task, and this is the evidence for it.

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

**Nine answers, not five.** `form` carries five variants whose field sets differ by
more than a name — onboarding asks for 22 fields, w4 for 9 — and the extractor selects
between them. A classifier that stops at `form` has answered half the question and left
the rest to the corpus, which is the hand-over this phase exists to remove. The label
set is generated from the type registry, so the classifier and the extractor cannot
disagree about what the possible answers are.

So the floor is `0.111`: nine balanced classes, always guessing one of them, having
read nothing. Every number below is reported against that.

### What a corpus rebuild cost, and why it was worth it

An earlier version of this section reported `0.958` on faxes and concluded that the
page image alone beat every model that also read the words. That result did not
survive its own corpus.

At the time, invoices and purchase orders had three page designs and two. Holding out
a *document* could not distinguish a model that had learned what an invoice is from one
that had memorised what this corpus's invoices look like — and the image model was
doing the second. Rebuilding the corpus with ten designs each and holding out a whole
*design* took the memorisation away:

| held out by | overall | fax | purchase orders |
|---|---|---|---|
| source document | `0.958` | `0.917` | `0.938` |
| **page design** | **`0.792`** | **`0.694`** | **`0.125`** |

Fourteen of sixteen purchase orders read as invoices. Not a data shortage — nine PO
designs were still in training — but a distinction the model never had to learn. An
invoice and a purchase order are both a header, a ruled line-item table and a totals
block; what separates them is a phrase printed at the top of the page. With two or
three templates per type, memorising each template stood in for the concept and looked
exactly like understanding.

### The pipeline: read the page, ask about the words only where it matters

Two models fail in different places rather than one being better. The image model is
right about almost everything except that one pair; a model given the words resolves
it outright. So the image model runs first and the text is consulted only where it is
known to be needed.

| unseen page designs, clean | image alone | cascade |
|---|---|---|
| purchase_order → invoice | 4 | **0** |
| invoice → purchase_order | 2 | **0** |
| **overall** | **`0.778`** | **`0.944`** |

The order is the whole economy of it. The image model reads no text, so it costs a page
render; the text path costs an OCR pass — hours over a thousand degraded documents.
Escalating spends that on the 22% of documents that need it rather than all of them.

The trigger is measured rather than chosen conservatively. The confusable-pair trigger
alone scores `0.944` and escalates 22%; adding a `0.90` confidence floor escalates 56%
and scores exactly `0.944`. Every extra escalation was a document the image model was
going to get right anyway, and each one is an OCR pass.

The keyword baseline is the secondary, and it is worth being clear about why that is
not a contradiction of calling it a floor elsewhere. It scores `0.700` overall and
`0.350` on multi-bill invoices, so it is no one's classifier. But asked a single
question — is this page an invoice or a purchase order, given that something else has
already narrowed it to those two — it is exact and free. A weak classifier can be a
strong arbiter.

### What the multi-bill invoice cost, and stopped costing

Multi-bill against plain invoice dominated Phase 1 and held the keyword baseline to
`0.350`. Across every run above it is now **zero confusions in either direction**. It
differs from an invoice by carrying a repeated per-service block, and a model reading
the page as a picture sees that structure instead of inferring it from vocabulary.

The confusion that replaced it was invoice against purchase order — invisible while
each type had two or three templates, and the worst class in the system once they had
ten.

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

### Abstention, and where it belongs now

Every error the image model made on unseen-design faxes arrived below `0.90`
confidence, and the documents it declined were the ones there was least to read on — a
mean of 43 OCR words against 92.5 on the ones it answered. It abstains where the page
has gone blank, which is the typed-decision discipline Phase 1 applied to fields,
arriving one stage earlier.

The cascade changes where that floor sits rather than whether it exists. A document
whose top two answers are the confusable pair is no longer a candidate for abstention —
it is a candidate for a second opinion, and the second opinion is exact. Abstention is
for pages nobody can read, not for questions that are merely narrow.

A document the pipeline declines is **not extracted and not graded**. Scoring an
abstention as a document whose every field came back empty would give it a zero and
make declining look identical to failing, which is the reverse of the truth. Coverage
is reported beside accuracy, and the declined files are listed in the run sidecar.

### Splitting: the free option won

A scanned batch does not arrive as one document per file, so the generator now builds
bundles — several documents concatenated, with the page each one starts on recorded.
120 bundles, 333 documents, 361 pages, and **half the joins are same-type on purpose**,
because one invoice following another is the join a change-of-type splitter cannot see.

| splitter | F1 | files exactly right | merged | over-cut |
|---|---|---|---|---|
| `single` — the file is one document | — | `0.108` | 213 | 0 |
| **`every_page`** — each page is a document | **`0.938`** | **`0.783`** | **0** | 28 |
| `by_type` — classify each page, cut on change | `0.772` | `0.458` | 62 | 27 |

The classifier-per-page splitter lost to cutting everywhere, and it is the Phase 2
cascade result again: the clever option beaten by the free one, visible only because
the baselines were reported beside it.

92% of the documents here are a single page, so cutting everywhere is wrong 28 times,
while `by_type` misses 62 same-type joins — `0.487` recall on exactly the joins it was
predicted to be blind to.

It also makes no merges, and a merge is unambiguously expensive: two unrelated
documents reach the extractor as one, and it emits a record for a document that never
existed.

An over-cut is not as cheap as it first looks, though, and running the pipeline end to
end is what showed it. A two-page multi-bill invoice cut in half produces a second half
that the classifier reads as a plain **invoice** — a header, a table and totals, with
the repeated per-service structure that distinguishes the type sitting on the page that
was cut away. So an over-cut does not merely lose fields; it can select the wrong
schema and fill it confidently. Both failure directions can invent, which is not what
this section originally claimed.

`by_type` stays in the codebase. It is the right shape wherever multi-page documents
are common — which this corpus is not — and running it is what produced the comparison.

### What classification costs the stage after it

The phase's own deliverable. Extraction over the same 175 documents, the same model and
prompts, one thing varying — where the document type came from:

| type from | field accuracy | exact match |
|---|---|---|
| the corpus | `0.924` | `0.780` |
| **the pipeline** | **`0.923`** | **`0.781`** |

One thousandth across 1,974 graded fields, and exact match a thousandth the other way.
The classifier placed all 175 correctly — type *and* variant — so the extractor received
the same schema either way, and what is left is run-to-run model variance rather than
classification error.

An earlier version of this table read `0.959` and `0.957` over 1,904 fields. Those were
the same two runs scored against the corpus as it stood *before* the rebuild to ten page
designs per type, and the report file holding them was never regenerated afterwards —
so a stale artifact was quietly overstating the pipeline by three and a half points.
Rescored against the corpus that exists, the level drops and the comparison the section
is about does not move: removing the answer key still costs a thousandth. Fixed when
Phase 5 went looking for a per-document score and found the totals disagreed.

So removing the corpus's answer key costs essentially nothing on clean documents, which
is the claim Phase 3 was built to test. Every extraction number before this one,
including Phase 1's, was produced with the corpus handing over the type.

Read it as production conditions on page designs the classifier has trained on. The
number that predicts a vendor template nobody has seen is the design-holdout `0.944`
above, not this one.

Visible in the same run and unrelated to classification: resumes score `0.820` against
`0.97` or better everywhere else, and `target_role` reaches `0.086` with 19 of 35
missing. The validators in Phase 4 land on the same documents from a different
direction — null employment years on every role — which is two instruments agreeing
about where the extractor is weakest.

### What this phase did not close

**Two form variants are still confused**, one document each: `loan` read as
`onboarding`, and `w4` read as `w9`. The keyword arbiter has no notion of variants, so
it cannot settle those the way it settles invoice against purchase order. They are the
next thing worth fixing and they are not what blocks anything.

**The models have not been compared across all four profiles on the rebuilt corpus.**
The one architecture comparison that has been run since the rebuild used clean
documents only — 36 of them, four per class — which establishes the mechanism and ranks
nothing. Whether a words-and-boxes model holds up on faxes, where docTR loses 38% of
the words, is the open question.

**Fields are not scored end to end through the splitter.** The chain runs — bundle to
split to classify to extract, with nothing handed the corpus's answers — but the
splitter is scored on boundaries and the extractor on documents, never on what comes
out of the far end of both. That needs a metric that aligns predicted pieces to true
documents and grades fields across the mismatch, and it is where the half-a-multi-bill
finding would appear as a number rather than as an anecdote.

**Bundles are built from clean documents only.** A real scanner batch is degraded, and
this corpus cannot yet ask whether splitting survives a fax.

## Phase 4: telling when it is wrong

The corpus has carried the ground truth for this since Phase 0 — 352 documents with 527
deliberately injected defects across 38 classes — so the stage can be scored rather than
admired.

The structural problem is that a validator runs on *extracted* output. When a rule
fires there are two explanations and they want opposite responses: the document is
defective, or the extractor misread a good one. Nothing in the firing tells them apart,
and a stage that cannot separate them reports a defect rate that is partly its own
extraction error and moves when the model changes.

So every rule is scored twice, and the first run is a gate rather than a result.

| scored against | precision | recall | document recall |
|---|---|---|---|
| the corpus labels — *does the rule work* | `0.911` | `0.918` | `0.974` |
| extracted output — *does the pipeline work* | `0.701` | `0.564` | `0.777` |

On ground truth there is no extractor to blame, so a rule that fires on a clean document
is simply wrong. That has to read zero before a rule ships — the same bar Phase 0 set by
demanding `score --predictions self` return exactly `1.000`. It does, so the gap between
those two rows has one attributable cause, and it is not the rules.

### What the gap is made of

`missing_bill_to` at `0.000`, `missing_vendor` at `0.154`, `missing_invoice_number` at
`0.455` — the extractor supplies a value the document does not have. That is Phase 1's
fabrication failure, still alive on every field nobody thought to mark optional, and the
validators find it without a label in sight.

`no_skills_listed` at `0.000` and both employment-date classes are resumes — the third
independent instrument to land on resumes, after the extraction score of `0.820` and the
`target_role` field at `0.086`.

**A validator firing because the extractor invented a value is not a false alarm. It is
a bug report.** That is the argument for the two-stage scoring, and it is why the
clean-corpus run matters: 24 of 175 documents with no injected defects were flagged, and
every one was a real extraction error — including a resume whose work history the labels
give as 2022 and 2021 and the extractor returned as null on every role.

### Two rules the self-test caught, and three fields that did not exist

`overlapping_service_periods` fired on 40 clean documents before it ever saw a
prediction. Services on one bill are normally billed over the same month, so overlap is
not a defect at all; what the generator injects is one section's period copied verbatim
onto another. Reading the injector rather than tuning the threshold took 40 false alarms
to 2 — and those last 2 are chance collisions on clean documents, so the rule is a
warning now. The self-test counts only errors: a rule that cannot be certain should not
be able to fail a build.

Three rules checked fields the schema does not have, and each failed the same silent
way — no field found, nothing compared, a clean `0.000` indistinguishable from the
defect being absent. Employment dates are `start_year`, not `start_date`. The claim
adjuster is `adjuster_name`. A W-9's TIN is `ssn` **or** `ein` depending on `tin_type`,
so it is not a check on any single field. A test now asserts every required field name
exists on the variant it is demanded of.

The largest class was worse than a wrong name: the signature block is printed on every
form and was recorded on none of them, living in the generator's render metadata rather
than its labels. 89 defects — the two biggest classes — with no ground truth to compare
against and no schema field to arrive in. Recording them moved document recall from
`0.883` to `0.974`.

### What this phase has not closed

`tax_miscalculated` is undecidable from the page: a tampered tax and a tampered total
both present as `subtotal + tax != total`, and nothing printed says which. Those
documents are still caught, under `total_mismatch` — which is why its precision reads
`0.345` while document recall holds. Wherever the corpus tags a cause the page cannot
distinguish, per-code recall understates the stage.

### What a validator is really measuring on a bad scan

The degraded set is derived from the clean corpus, so it carries no injected defects at
all. Every finding on it is a false alarm by construction, and the rate is the answer to
how fast extraction error turns into phantom defects:

| profile | documents | flagged | false-alarm rate |
|---|---|---|---|
| clean | 175 | 24 | `0.137` |
| light | 76 | 10 | `0.132` |
| photo | 51 | 12 | `0.235` |
| **fax** | **48** | **36** | **`0.750`** |

Three out of four clean faxes are reported as defective. The rules did not change and
the self-test still says they are correct — on a fax they are measuring OCR quality
wearing a validator's name. The codes confirm it: `line_item_math_error` ×29 and
`section_line_item_math_error` ×25, which is what happens when digits are misread and
the arithmetic stops footing.

Light degradation costs nothing at all — `0.132` against clean's `0.137`. The damage is
not gradual; it arrives with the fax.

This lands squarely on Phase 5. **You cannot route on validator findings for fax
documents**: doing so sends three quarters of the good ones to a person. A defect
signal is only actionable where the extraction under it is trustworthy, which is the
same shape as Phase 2's lesson about confidence — a signal is useful only where it is
calibrated, and the place you most want to trust it is the place least likely to
deserve it.

**Nothing routes on a finding.** `Report.ok` exists and no stage consumes it. Confidence
and routing are Phase 5, and smearing them into Phase 4 would make both unmeasurable.

## Phase 5: is the confidence real

The phase asks one question — when the pipeline is wrong, did it know? — and the answer
turned out to be no for the signal everybody reaches for first.

### The curve, and the baseline that makes it mean anything

A coverage curve drawn alone always looks like an achievement. Raise a confidence floor,
watch accuracy climb on what remains, and conclude the confidence is working. It climbs
whether or not the confidence carries information, because you are also throwing
documents away.

So every curve here is drawn against **random abstention**: decline the same *number* of
documents, chosen at random. That baseline needs no sampling — the expected accuracy of
a uniformly random subset is the accuracy of the whole — so it is a flat line at the
corpus mean, and the vertical distance to it is the entire value of sorting by
confidence. Where the curve sits on the line, the floor is buying nothing.

### Two floors that were not doing their job

Building the measurement found both.

The manifest carried `[classifiers.dit] abstain_below = 0.9`, and under a cascade it did
nothing at all. The cascade builds its members with abstention disabled on purpose — a
primary that declines internally hands back no answer and no runner-up, leaving the
escalation nothing to arbitrate — so the floor had to be declared on the cascade or not
at all. And the cascade applied its own floor on only one of its two exits, the
escalation path, so it governed the minority of documents the primary was already unsure
about. Exactly the wrong half. Nothing anywhere reported a floor that was not being
applied.

Second, abstaining blanked the predicted type and dropped the answer with it. A declined
document recorded no answer, which makes *is this floor set right?* unanswerable from the
artifacts — the coverage curve could only ever be drawn above whatever floor was already
in force. Classifications now carry `withheld`, the answer that was suppressed.

### Where the floor belongs

Measured on the cascade that actually runs, over documents held out **by page design** —
the split that does not let a model recognise a layout it trained on:

| floor | coverage | accuracy | errors through |
|---|---|---|---|
| `0.60` | `0.771` | `0.946` | 6 |
| `0.70` | `0.618` | `0.978` | 2 |
| **`0.85`** | **`0.472`** | **`1.000`** | **0** |
| `0.90` | `0.438` | `1.000` | 0 |

`0.85`, not `0.90`: they answer with the same zero errors and `0.85` answers five points
more of the corpus. The honest reading of `0.472` is that on page designs it has never
seen, this classifier can be trusted unsupervised on slightly under half the corpus.

The same run answered a question nobody had asked. Against its own primary at matched
coverage, the cascade's text arbiter gains `+0.081` at a `0.50` floor, `+0.036` at
`0.60`, and **nothing at all from `0.75` upward**. Every document it rescues sits below
`0.75`. At the floor now set, the cascade and bare DiT are the same pipeline and the OCR
is spent for nothing — so `0.944` is not an argument for the escalation at every
operating point, only at low ones.

### The confidence is real, and it is about the wrong thing

The classifier is well calibrated for classification: ECE `0.063` on the design holdout.
Then the same confidence was scored against whether the *extraction* came back right —
which is what a floor actually decides, since a floor sends a document to a person and a
person cares about the fields.

It is worse than uninformative. Pooled over 175 documents, raising the floor makes the
accepted half **worse**: `0.917` at no floor, `0.888` at `0.95`, every row at or below
the random baseline.

Split by type, it is a confound rather than a broken model:

| type | mean confidence | field accuracy |
|---|---|---|
| resumes | `0.997` | `0.820` |
| multi-bill invoices | `0.994` | `0.976` |
| forms | `0.993` | `0.840` |
| purchase orders | `0.874` | `0.971` |
| invoices | `0.872` | `0.979` |

DiT's confidence measures how visually distinctive a page is. Extraction difficulty is
driven by field count and free text. On this corpus those run opposite, so the pooled
curve averages across the variable driving both columns. The scorer now reports a
per-type breakdown and says so in words when the ordering inverts — a tool that reported
only the pooled number would hand you *the model is broken* when the answer is *you are
looking at two populations*.

### What does predict a bad extraction

Signals that are observations about the document rather than the model's opinion of its
own work. On 1,055 degraded documents across all four profiles, routing the least
promising 20% to a person, measured against the same random baseline:

| signal | rank correlation | lift |
|---|---|---|
| **share of fields returned blank** | `-0.786` | **`+0.120`** |
| words per page | `0.512` | `+0.107` |
| mean OCR word confidence | `0.697` | `+0.105` |
| validator errors | `-0.494` | `+0.087` |
| *classifier confidence* | `0.240` | `+0.028` |
| validator warnings | `-0.014` | `+0.001` |
| OCR engine disagreement | `0.044` | `-0.018` |

The classifier's own confidence is in that table deliberately, scored on the same
documents by the same method at the same fixed coverage — measured on separate corpora
it would be an anecdote. **The best observable signal is worth four times the model's
self-report**, and it is the best signal inside every document type as well as pooled,
which is the test the classifier's confidence failed.

Three results worth keeping:

**Phase 4's severity split paid for itself here.** Validator *errors* carry real signal
and *warnings* carry none. Summing them — the obvious thing to do — would have diluted
one into the other.

**Engine disagreement was expected to help and does not.** Two OCR engines differing
about how much text is on a page turns out not to predict whether the fields came back
right. Directions are declared before looking, so this is recorded as a contradicted
expectation rather than quietly re-read; given ten signals one will point the right way
by chance, and letting the data choose each sign is how noise becomes a finding.

**Routing helps least where it is needed most.** On fax the baseline is `0.253` and the
best signal reaches `0.333`; on light it is `0.894` with no room to move. Signals sort
documents within a difficulty band. They do not rescue one.

### Routing

`router` is a plugin slot between validator and sink. Independent gates, not one blended
score — a wrong type is not a bad field, because the type chose the schema, and one
number would let a confident classification vouch for a hopeless extraction.

Gates rather than a fitted model for three reasons. A model would need its own
train/test split; it would relearn the document-type confound that already produced one
wrong conclusion here; and a queue entry has to say *why*. It says
`blank_share 0.75 above 0.2; validator_errors 2 above 0`, which a reviewer can check and
disagree with.

On the 1,055 degraded documents: accepts `0.337` of them at `0.846` mean field accuracy
against a `0.552` corpus baseline — a lift of `+0.293`. All three gates fire as the sole
reason for a decision (131, 125 and 52 times), so none is carried by its neighbours.

And the measurement that argues against part of its own configuration: on the clean run
the classifier gate fires 23 times, **every time as the only gate**, on documents that
extracted at a mean of `0.992`. On designs the model has seen, that gate is almost pure
cost. It is kept at `0.85` because the deployment assumption is vendor templates nobody
trained on — the case the number was measured for — and the cost is written next to the
setting rather than tuned away.

### What this phase did not close

**No signal was fitted, only measured.** Each gate is a threshold on one signal. A model
combining them would score better on this corpus, and the reason not to ship one is
above; the reason to revisit it is that `blank_share`, `words_per_page` and
`ocr_confidence` are substantially reading the same underlying fact from different ends.

**The extraction curve is clean-corpus only.** Confidence against extraction was scored
on 175 clean documents, because that is the run whose signals could be reconstructed.
The equivalent on degraded documents needs a re-run.

**Nothing consumes the queue.** `route.cli apply` writes it; no interface reads it, and
no correction flows back. That is Phase 7.

## Phase 6: can it repair itself

The phase asks whether a second attempt at a document helps. The answer depends
entirely on whether the second attempt is anchored to the first — and the phase spent
most of its effort discovering that its own instruments were lying to it.

### The scorer was written first, and adversarially

Repair is the first stage here that can raise its own score by making documents worse.
It is triggered by complaints, so the obvious success metric is "did the complaints
stop" — and the cheapest way to stop an arithmetic complaint is to return an empty
`tax_amount`. Every dashboard would show it working.

So `eval/repair.py` was written and committed before any loop existed. Success is
measured against the corpus labels and never against the gates; gate clearance is
reported *beside* the real number, because the distance between them is the diagnostic
for a loop silencing its critics. `damaged` sits next to `improved` everywhere, with a
Wilson interval, because a rate observed on forty documents is an estimate that reads
like a fact. A crashed call leaves the document at `before`, never at zero — an outage
must not read as a damaging loop.

### Three arms, and the middle one is the point

    no_repair    the original extraction, untouched
    rerun        the IDENTICAL request again — no complaints, no previous answer
    reprompt     the same request with the complaints quoted back

`rerun` is what makes the phase measurable. The extractor is sampled, so a second
request improves some documents by luck that *any* repair inherits for free. Without
that arm, a guided repair reporting `+0.058` looks like the feedback working when much
of it is sampling temperature.

Every arm starts from the identical frozen extraction, so no extraction variance enters
before the experiment starts. Guided arms iterate — attempt N sees attempt N-1's record
and the complaints recomputed against it — and blind arms repeat the identical request.
Arms are compared only at equal call budgets, because three guided attempts against one
blind sample prices the extra sampling as though it were the guidance.

### The result

| corpus | baseline | blind vs nothing | guided vs nothing | guided vs blind |
|---|---|---|---|---|
| clean | `0.881` | **`+0.043`** [+0.024, +0.065] | **`+0.058`** [+0.039, +0.079] | `+0.015` spans zero |
| degraded | `0.303` | **`-0.010`** [-0.019, -0.001] | `+0.002` spans zero | **`+0.012`** [+0.001, +0.023] |

Read across the rows.

**On clean documents a second pass helps, and it barely matters whether it is guided.**
Both arms are resolvably better than doing nothing; the difference between them is not
resolvable and would need about 356 documents to settle. Zero documents were damaged in
either arm.

**On degraded documents, blind resampling is harmful and guidance prevents that.** The
blind arm is resolvably worse than not running at all. The guided arm is not
distinguishable from doing nothing in either direction — it has *not* demonstrated
positive lift — and it resolvably beats the blind arm. That last comparison is the only
resolvable thing on that row.

Stated carefully, because the loose version is tempting and wrong: repair is clearly
beneficial on clean documents; on degraded documents blind retrying is harmful, guided
repair avoids most of that damage, and guided repair has not been shown to help.

### What the guidance actually does

The field transitions say it more plainly than the deltas.

| | fields repaired | fields damaged |
|---|---|---|
| clean, blind | 29 | **0** |
| clean, guided | 34 | **0** |
| degraded, blind | 20 | **52** |
| degraded, guided | 7 | **5** |

The blind arm on degraded documents *repairs nearly three times as many fields as the
guided one*. Its problem is not that it cannot fix things — it is that it destroys 52 to
do it, mostly correct values replaced with wrong ones (22) or dropped entirely (27).

So the guidance's value is **conservation, not correction**. It does not make the model
better at reading the page; it gives it a reason to keep the answer it already had. The
sentence doing the work is probably not the complaint list but *"If, after re-reading,
you believe your previous answer was right, return it unchanged."*

That unifies the two rows. On clean text the answer is largely determined, so a second
draw lands near the first and resampling converges. On ruined text the answer is
underdetermined, the first answer was partly right by luck, and an independent draw
discards that luck. **Resampling helps where the answer is determined and hurts where it
is not** — a property of any loop that re-does work, not a fact about OCR.

### The bug that produced a headline, and then removed it

The first version of this section reported that repair invented `business_name` on 48 of
48 eligible W-9s, in both arms — a hundred per cent failure on one field — and explained
it as a named slot in a schema exerting more pressure on a second pass than an
instruction not to fill it. It was written up as a result.

It was a bug in the repair runner. Optional fields are asked as a *decision* rather than
a slot: the model answers `{"status": ..., "value": ...}` and the extractor flattens that
with `collapse_optional` before anything else sees it. The repair merge reimplemented
merging and never called it, so a repaired record kept the dict, and the scorer comparing
a dict against a blank truth read it as a fabricated value. Asked directly, the model had
answered `"unclear"` on 47 of the 48. That collapses to absent. It was **correct**.

Two optional fields exist in the entire schema registry. The transition table showed
damage of exactly 48 and 5 — fifty-three, every one of the guided arm's inventions.

Fixing it inverted the phase. Guided repair on degraded documents moved from `-0.029` to
`+0.002` and from 56 damaged documents to 5. The Goodhart risk ratio — the headline
diagnostic, "damaged documents were 4.43× more likely to have their gates go quiet" —
became `1.638` with an interval of `[0.917, 2.924]`, spanning 1. The effect was almost
entirely the bug.

Three things worth keeping from it.

**The per-field table found its own scorer's bug.** A document-level `-0.029` was
unfalsifiable at that resolution and would have shipped. Only concentrating the damage
into one field on one variant made it checkable, and obviously wrong once checked. The
diagnostic built to catch the model gaming the metric caught the harness instead.

**Two implementations of one rule, five times in one phase.** `repair.cli` against
`route.cli` on which documents were flagged. The per-document scorer against the
aggregate. The repair merge against the extractor's. A PaddleOCR detection region
against a `Word`. And `--limit 15` today against `--limit 15` historically, which
silently selected a different 75 documents. Every one was an implicit equivalence
assumed rather than asserted; every one was plausible; none broke visibly.

**A finding that concentrates suspiciously is a finding to check.** 48 of 48 is not what
model behaviour looks like. It was written up instead of checked, and the check took
ninety seconds.

### What a corrected result is worth

`core/stamp.py` now writes provenance into every report, because twice in this phase the
expensive part of a bug was not fixing it but working out which artifacts had inherited
it. Four things move independently and a hash of one says nothing about the others: the
code, the *meaning* of the metric (a hand-bumped `evaluation_version`, since a commit
says the code differed and not whether the difference mattered), the corpus in two senses
— labels for "did the expected answers change" and document bytes for "did the system see
different pixels" — and the cohort actually evaluated.

The cohort hash earned itself immediately. Three different "75-document sets" are in
circulation in this repository, overlapping each other on 1 and 52 documents. A
comparison across two of them would have looked entirely normal.

### What this phase did not close

**Tool-using extraction was not built.** The phase's wording covers the bounded repair
loop *and* tool use; only the loop exists, and the loop is what produces the phase's
stated number. On this corpus a tool would likely buy little: the dominant failures are
perceptual — fax OCR at `0.305`, characters that were never produced — and
schema-comprehension, not arithmetic, and the validators already detect the arithmetic.

**Selection is the strongest remaining question and the smallest remaining prize.** With
an oracle choosing which documents to repair, guided repair on degraded documents would
move from `+0.0022` to `+0.0050` — the entire ceiling is not damaging five fields across
two hundred documents. As a research question — can you predict when generative revision
has positive expected value, from features available *before* revision — it is genuinely
interesting and portable. As an accuracy lever on this corpus it is worth three tenths of
a point, against `+0.315` for choosing a better OCR engine on photographs.

**Where the accuracy actually is.** Fields the extractor never reads in either direction:
`business_name` scored 0 of 12 where the document carried one, `target_role` reaches
`0.086`. Those are schema and prompt problems, and they are worth more than any
refinement of repair.


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
