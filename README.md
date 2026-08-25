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
tools/document-generator/   synthetic evaluation corpus (Docker; see its README)
data/                       generated corpus — gitignored, regenerable from a seed
docker-compose.yml          services; the generator is on-demand tooling
```

Application packages (`core/`, `plugins/`, `web/`, `worker/`, `eval/`, `packs/`) land
here as they are built. They are deliberately not scaffolded yet — empty directories
are not structure.

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

The generator is behind the `tools` profile, so `docker compose up` ignores it. It is
on-demand only; `run --rm` starts it for one job and throws the container away.

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
