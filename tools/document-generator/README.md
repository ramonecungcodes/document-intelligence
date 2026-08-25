# di-document-generator

Builds the synthetic evaluation corpus for DocumentIntelligence: realistic
**invoices, purchase orders, resumes, forms and multi-bill invoices** as PDFs, each
paired with **ground-truth JSON labels** so extraction accuracy can be measured
field by field.

Runs in Docker. Nothing needs to be installed locally -- no Python, no Chrome.

```bash
# from the repo root
docker compose run --rm --name di-document-generator document-generator build
docker compose run --rm --name di-document-generator document-generator build --degrade --levels light,medium,heavy,photo,fax
```

Output lands in `./data` on the host. See "Running it" below for the full command set.

Everything here is fabricated — every name, company, address, and phone number is
invented — so it is safe to share, commit, or publish.

---

## What it produces

Five document types, each in several structural layouts with their own fonts and
brand colours, so a model is tested on formats it has not seen rather than one rigid
template. Counts are set by flags (`--invoices-per-company`, `--multibill`, ...);
`build` with no arguments produces 352 PDFs.

| Type | Varies by |
|------|-----------|
| Invoices | 3 layouts × 7 vendor brands |
| Purchase orders | 2 layouts × 7 brands |
| Resumes | 8 layouts × 4 roles (management, developer, HR, RPA developer) |
| Forms | 3 layouts × 5 kinds (onboarding, insurance claim, W-9, W-4, loan) |
| Multi-bill invoices | 3 layouts × 5 vendors, 2–4 separately-payable services each |

Each type is written to its own directory under the corpus root (`/data` in the
container, `./data` on the host), with ground truth in `labels/<type>.json`. Every
label carries a `file` field pointing at its PDF, so model output joins to truth on
that key. `--irregular` writes a parallel defective corpus, and `degrade` writes
image-only scans; both mirror the same layout.

## Samples

Six documents are committed under [`samples/`](samples/) so the schemas below can be
read against real output without building anything:

| Sample | Shows |
|--------|-------|
| `invoice.pdf` | the baseline single-vendor case |
| `multi-bill-invoice.pdf` | 3 separately-payable services, each with its own account number and total |
| `multi-bill-invoice.defective.pdf` | `missing_section_account` + `section_total_mismatch`, both visible on the page |
| `multi-bill-invoice.scanned.pdf` | the same document at `medium` degradation — **no text layer**, so it forces OCR |
| `form-w9.pdf` | a government form with checkbox and TIN fields |
| `resume.pdf` | free-form layout with nested work history |

Each `.pdf` sits next to its `.json` ground truth, lifted verbatim from the corpus
labels; the only field altered is `file`, repointed at the sample's own name.

## Label schemas

> Every label (all types) also includes an **`irregularities`** array — empty `[]`
> for this clean set, and populated with defect tags in the separate `irregular/`
> set (see "Irregular (defective) documents" below).

**`labels/invoices.json`** — array of:
```json
{
  "file": "invoices/<name>.pdf", "doc_type": "invoice",
  "vendor_name": "...", "invoice_number": "INV-...",
  "invoice_date": "YYYY-MM-DD", "due_date": "YYYY-MM-DD",
  "po_number": "PO-...", "bill_to": "...", "terms": "Net 30",
  "currency": "USD", "subtotal": 0.0, "tax": 0.0, "total": 0.0,
  "line_items": [{"description": "...", "quantity": 0, "unit_price": 0.0, "amount": 0.0}]
}
```

**`labels/purchase_orders.json`** — array of:
```json
{
  "file": "purchase_orders/<name>.pdf", "doc_type": "purchase_order",
  "buyer": "...", "vendor": "...", "po_number": "PO-...",
  "po_date": "YYYY-MM-DD", "delivery_date": "YYYY-MM-DD",
  "terms": "Net 30", "currency": "USD",
  "subtotal": 0.0, "tax": 0.0, "total": 0.0,
  "line_items": [{"description": "...", "quantity": 0, "unit_price": 0.0, "amount": 0.0}]
}
```

**`labels/resumes.json`** — array of:
```json
{
  "file": "resumes/<name>.pdf", "doc_type": "resume", "layout": 0,
  "name": "...", "email": "...", "phone": "...", "location": "City, ST",
  "target_role": "RPA Developer", "current_title": "...",
  "years_experience": 0, "skills": ["..."], "certifications": ["..."],
  "education": {"degree": "...", "school": "...", "year": 0},
  "work_history": [{"company": "...", "title": "...", "start_year": 0, "end_year": "Present"}]
}
```

**`labels/forms.json`** — array; every entry has `file`, `doc_type:"form"`,
`form_type`, `layout`, plus fields that depend on `form_type`:

| form_type | key fields |
|-----------|-----------|
| `onboarding` | employee_name, ssn, date_of_birth, home_address, personal_email, phone, job_title, department, manager, start_date, employment_type, pay_rate, pay_frequency, bank_name, bank_routing, bank_account, w4_filing_status, allowances, emergency_contact_name, emergency_contact_phone, i9_verified (bool), handbook_acknowledged (bool) |
| `claim` | claim_number, policy_number, claimant_name, claim_type, date_of_loss, date_reported, incident_location, loss_description, claim_amount, deductible, adjuster_name, status, contact_phone |
| `w9` | name, business_name, tax_classification, address, city_state_zip, tin_type (`SSN`\|`EIN`), ssn, ein, requester |
| `w4` | name, ssn, address, filing_status, multiple_jobs (bool), dependents_amount, other_income, deductions, extra_withholding |
| `loan` | application_number, applicant_name, ssn, date_of_birth, address, phone, email, employer, job_title, years_employed, annual_income, loan_type, loan_amount, loan_term_months, loan_purpose, down_payment, monthly_debt, credit_score, co_applicant_name |

**`labels/multi_bill_invoices.json`** — array of:
```json
{
  "file": "multi_bill_invoices/<name>.pdf", "doc_type": "multi_bill_invoice", "layout": 0,
  "vendor_name": "...", "invoice_number": "CMU-...", "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD", "bill_to": "...", "master_account": "...",
  "terms": "Net 30", "currency": "USD", "section_count": 3,
  "sections": [
    {
      "section_index": 0, "service_type": "Water Service", "service_code": "WTR",
      "account_number": "UTL-201414",          // the key AP splits on
      "reference_label": "Meter", "reference_number": "M7022674",
      "service_location": null,                 // set for by-site / by-shipment vendors
      "cost_center": "CC-6180 Manufacturing",
      "service_period_start": "YYYY-MM-DD", "service_period_end": "YYYY-MM-DD",
      "line_items": [{"description": "...", "quantity": 0, "unit_price": 0.0, "amount": 0.0}],
      "subtotal": 0.0, "tax": 0.0, "total": 0.0
    }
  ],
  "subtotal": 0.0, "tax": 0.0, "total": 0.0
}
```

### Why multi-bill invoices are their own type

A utility bills water and gas on one document. A carrier bills three shipments. A
landlord bills four sites. The vendor issues **one** invoice number, but accounts
payable has to raise a **separate internal invoice per service** so each can be coded
and paid on its own — which means extraction has to return a repeating group, not a
flat record, and each group needs the identifier the payment will be routed by.

This is a different problem from splitting a PDF into several documents. There is one
document, often one page; the split is *inside* it. An extractor that returns a single
`total` for these is wrong even when the number is right.

What the set exercises:

- **Repeating-group extraction** — 2–4 sections per invoice, 114 sections across the 40
  clean documents, in three layouts: summary-table-then-detail, boxed per-service cards,
  and one continuous ledger with grouped bands.
- **Roll-up validation** — `sum(sections[].total) == total`, and the same for subtotals.
  Every section also carries its own tax rate, so the tax cannot be derived invoice-wide.
- **Routing keys** — each section has its own `account_number`, `reference_number`
  (meter / circuit / BOL / route / contract) and `cost_center`. Sections whose account
  numbers are missing or duplicated cannot be paid separately, which is a real failure.
- **Per-section service periods** — meters get read on different days, so the periods
  differ per section and may not be derivable from the invoice date.

### Building validation rules against the forms

The form fields are deliberately typed so you can write and test validation rules:

- **Format rules:** SSN `XXX-XX-XXXX`, EIN `XX-XXXXXXX`, bank routing = 9 digits, email pattern, phone pattern.
- **Enum rules:** `employment_type`, `w4_filing_status`/`filing_status`, `claim_type`, `status`, `loan_type`, `tax_classification` all draw from fixed value sets.
- **Range/numeric rules:** monetary fields ≥ 0; `credit_score` in 300–850; `loan_amount` > 0; `allowances` ≥ 0.
- **Cross-field rules:** `date_reported` ≥ `date_of_loss`; `date_of_birth` implies age ≥ 18 vs. `start_date`/today; W-9 has **exactly one** of `ssn`/`ein` (governed by `tin_type`); loan `down_payment` ≤ `loan_amount`.
- **Required/boolean rules:** checkbox fields (`i9_verified`, `handbook_acknowledged`, `multiple_jobs`) are booleans; required-field presence.

Dates are stored ISO (`YYYY-MM-DD`) in the labels but rendered `MM/DD/YYYY` on the
document, so extraction has to normalize — a realistic validation concern.

---

## How to use it

1. Run your extractor (LLM/agent, OCR, whatever) over the PDFs in
   `invoices/`, `purchase_orders/`, `resumes/`, `multi_bill_invoices/`.
2. Load the matching `labels/*.json` and join on the `file` field.
3. Compare predicted vs. expected per field. Suggested metrics:
   - **Scalar fields** (numbers, dates, names): exact-match accuracy, plus
     precision/recall/F1 across the field set.
   - **Line items / work history / invoice sections** (lists): match rows, then score
     field-level precision/recall so partial extractions are graded fairly. For
     multi-bill invoices, score section *recall* separately — missing a whole billable
     service is a different failure from getting one of its fields wrong.
   - Track **cost and latency per document** alongside accuracy.

Because the data is fully labeled, you can report a real number like
"invoice header-field accuracy 0.94, line-item F1 0.88" instead of eyeballing a demo.

---

## Running it

Every command is `docker compose run --rm --name di-document-generator document-generator <subcommand>`,
run from the repo root. Abbreviated below as `<gen>`.

| Subcommand | Does |
|------------|------|
| `build` | clean set + defective set, both rendered to PDF |
| `build --degrade --levels ...` | the above, plus the scanned variants |
| `generate` | HTML + labels only |
| `render` | HTML → PDF |
| `degrade` | PDF → image-only scans |
| `help` | usage and the resolved dataset root |

```bash
<gen> build                                        # the standard corpus
<gen> build --degrade --levels light,medium,heavy,photo,fax
<gen> build --multibill 60 --forms-per-type 60     # more of a given type
<gen> degrade --out /data/irregular --levels fax   # one stage on its own
```

Flags after the subcommand pass straight through to the underlying script, so
`--seed`, `--out`, `--only`, `--force`, `--limit` and the per-type counts all work.

Paths given to the container are container paths: the corpus root is `/data`, which
is `./data` on the host.

> **On Git Bash / MSYS (Windows):** an absolute container path passed as an argument
> gets rewritten to a Windows path before Docker sees it, so `--out /data/irregular`
> silently becomes something like `/tmp/C:/Program Files/Git/data/irregular` and the
> command reports `0/0` files. Prefix the command with `MSYS_NO_PATHCONV=1`, or use
> PowerShell. Subcommands that take no path argument are unaffected.

### A held-out set the model has never seen

The generator is **seeded**: the same seed reproduces the same corpus exactly; a new
seed produces brand-new documents and data. Keep a separate evaluation set so you
never measure on documents you tuned against.

```bash
<gen> generate --seed 99 --out /data/heldout
<gen> render   --out /data/heldout
<gen> degrade  --out /data/heldout --levels medium,photo
```

Recommended split: **seed 42 for development, a different seed for final evaluation.**

### Running it without Docker

The scripts are plain Python and still work directly. `generate.py` and
`render_pdfs.py` need only the standard library plus Chrome or Edge on PATH;
`degrade.py` needs `pip install pymupdf pillow augraphy`. Set `DI_DATASET_ROOT`
to choose where the corpus goes, or pass `--out`.

## Irregular (defective) documents

A parallel dataset of the **same document types with injected defects** — for
testing validation rules, error detection, and the human-in-the-loop review path.

Generate it (writes to a separate `irregular/` folder by default):

```bash
<gen> generate --irregular
<gen> render --out /data/irregular
```

`build` does both sets already; the above is for regenerating just the defective one.

Every label carries an **`irregularities`** array listing the defect tags injected
into that document (`[]` for the clean set). Each irregular document has 1–2
defects. Because a defect mutates the document's underlying data, the PDF and its
label stay consistent — e.g. a `total_mismatch` invoice *shows* the wrong total and
the label *records* that wrong total plus the `total_mismatch` tag, so you can score
both extraction (did you read the value that's there?) and detection (did you flag
the problem?).

**Defect catalog**

- **Invoices / POs:** `total_mismatch`, `subtotal_mismatch`, `tax_miscalculated`, `line_item_math_error`, `negative_line_amount`, `missing_invoice_number` / `missing_po_number`, `missing_bill_to`, `missing_vendor`.
- **Resumes:** `missing_email`, `missing_phone`, `no_skills_listed`, `missing_employment_dates`, `impossible_employment_dates`.
- **Multi-bill invoices:** `section_total_mismatch`, `invoice_total_not_sum_of_sections`, `duplicate_section_account`, `missing_section_account`, `missing_section_period`, `overlapping_service_periods`, `negative_section_total`, `section_count_mismatch`, `section_line_item_math_error`, `missing_invoice_number`, `missing_bill_to`.
- **Forms:** `missing_signature`, `missing_sign_date`, `invalid_ssn_format`, `invalid_routing_number`, `missing_bank_account`, `no_filing_status_selected`, `no_tax_classification_selected`, `no_tin_type_selected`, `missing_tin`, `date_reported_before_loss`, `negative_claim_amount`, `missing_adjuster`, `credit_score_out_of_range`, `negative_income`, `down_payment_exceeds_loan`.

Use it two ways:
1. **Extraction robustness** — does the model still pull the right (even if wrong) values off a messy document?
2. **Detection** — do your validation rules flag exactly the injected defects? Score detection precision/recall against the `irregularities` ground truth.

---

## Scanned and photographed documents

Every PDF here has a perfect text layer, so an extractor can read it without ever
touching OCR — which makes accuracy numbers look better than they will be in
production. `degrade.py` rasterises the PDFs and puts the images back through a
print/scan pipeline, writing **image-only documents with no text layer**. That
forces the OCR path and gives you an honest number.

```bash
<gen> degrade                                     # /data -> /data/degraded (medium)
<gen> degrade --levels light,medium,heavy         # three variants per document
<gen> degrade --out /data/irregular               # degrade the defective set too
<gen> degrade --only invoices --levels photo      # phone-photo invoices only
<gen> degrade --format png                        # images instead of image-only PDFs
```

Or in one shot with everything else: `<gen> build --degrade --levels medium,photo`.

**Profiles**

| Profile | dpi | Looks like |
|---------|-----|-----------|
| `light`  | 200 | a crisp office scanner; barely degraded |
| `medium` | 150 | a typical scan: grain, paper texture, slight skew, uneven light |
| `heavy`  | 120 | a bad scan: low ink, dirty drum, flat contrast — readable, but only just |
| `photo`  | 150 | a phone snap: perspective, a cast shadow, warm colour, uneven light |
| `fax`    | 170 | photocopied then faxed: bilevel, roller streaks, blown-out small type |
| `mixed`  |  —  | one of the above picked at random per document |

Every profile is tuned to stay **legible**. A document no human can read teaches you
nothing about OCR quality — it only guarantees failure.

**Ground truth comes along unchanged.** Each degraded document keeps the label of the
document it came from, `irregularities` included, so the clean set and the defective
set both degrade with no re-labelling. Labels are written to `degraded/labels/*.json`
with two fields added:

```json
{
  "file": "invoices/northwind_INV-20261000__medium.pdf",
  "source_file": "invoices/northwind_INV-20261000.pdf",
  "degradation": {
    "level": "medium", "profile": "medium", "engine": "augraphy",
    "dpi": 150, "text_layer": false,
    "ops": ["PerspectiveWarp", "Skew", "InkBleed", "SubtleNoise", "..."]
  }
}
```

Because `degradation.profile` is recorded per document, you can report accuracy
**per profile** rather than as one blended number — which is what shows where your
OCR/normalisation actually breaks, rather than just that it does.

**Reproducible.** Degradation is seeded per `(document, level)`, so the same `--seed`
reproduces the same scans byte for byte, and adding documents never reshuffles the
existing ones. Use a different `--seed` alongside a different generator seed for a
held-out set.

**Engines.** `--engine augraphy` (default) uses [augraphy](https://github.com/sparkfish/augraphy)
for genuine print/scan physics — ink bleed, low-ink lines, dirty drum, lighting
gradients, faxification. `--engine builtin` needs only pymupdf and pillow and
approximates the same effects; useful if you would rather not install augraphy.
Geometry (skew, perspective) is done with Pillow in both, because augraphy's
`Geometric` only accepts whole-degree rotation and most crooked scans are under 2°.

---

## Requirements

Docker, and nothing else. The image is `python:3.12-slim-bookworm` plus:

- **chromium** — renders the HTML to PDF
- **libgl1 / libglib2.0-0** — augraphy pulls in `opencv-python`, which links against them
- **fonts-liberation, fonts-dejavu-core, fonts-crosextra-carlito, fonts-crosextra-caladea** —
  the documents ask for Georgia, Calibri, Cambria, Arial and Courier New, none of which
  ship on Linux. These are the metric-compatible substitutes; without them Chromium
  collapses every face to one fallback and the layout variety is lost.
- **pymupdf, pillow, augraphy, numpy** — the degradation pipeline

Built image: ~2.4 GB.

### Rendering fidelity vs. a Windows host

Labels are produced by seeded Python and are **byte-identical** on any platform. The
rendered PDFs are not quite: Calibri, Cambria, Arial and Courier New substitute exactly,
but **Georgia has no free metric-compatible clone** and falls back to DejaVu Serif, so
Georgia documents are visually a little different from a Windows-Chrome render. Layout,
pagination and content are unaffected. Pick one platform and regenerate the whole corpus
there rather than mixing renders across platforms.

## Notes

- All content is synthetic. Any resemblance to real people or companies is coincidental.
- The 7 invoice/PO companies are consistent across runs of the same seed; resume
  candidates are generated fresh from name/skill/role pools.
- PDFs are produced by rendering HTML in headless Chrome, so they carry a real
  text layer (selectable/extractable), not scanned images. To test the OCR path,
  use `degrade.py` — see "Scanned and photographed documents" above.
- augraphy writes a ~9 MB texture cache when it runs. In the container it goes to
  `/tmp`, so it never reaches the mounted volume. Running the scripts directly on a
  host will drop an `augraphy_cache/` next to them; it is gitignored.
- `data/` is gitignored in full. The corpus is regenerable from a seed, so it is never
  committed.
- `samples/` is the exception and *is* committed: six documents, ~700 KB, so the label
  schemas above can be checked against real output without building anything.
