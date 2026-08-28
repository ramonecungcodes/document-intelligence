"""A classifier that reads where the words are, not only what they say.

The text classifiers in this package see a string. On a clean document that is plenty
-- the LLM scores 0.990. On a fax it is not: docTR finds only 62% of the words on a
170 dpi bitonal page, and no amount of comprehension recovers a word that was never
read. That is the whole of the 0.990-to-0.571 gap.

Geometry survives what glyphs do not. Measured before this was built, on the same
faxes: a coarse ink-occupancy grid with no words in it at all, nearest-neighboured
against the clean corpus, scored 0.821. The ink is still in the right places long
after it has stopped being legible.

So this one is handed the words, a box for each of them, and a picture of the page.
The distinction it should win on is multi-bill invoices, which say "Invoice" exactly
as loudly as invoices do and differ only in carrying a repeated per-service block --
a fact about structure that a text classifier has to infer and this one can see.

It needs a fine-tuned checkpoint; the pretrained base has no notion of these five
types. `tools/train-layout-classifier.py` produces one, and refuses to guess when the
directory named here does not contain it -- silently falling back to the base model
would produce confident nonsense rather than an error.
"""
from __future__ import annotations

import os
import time

from classify.base import Classification, register, split_label
from core.plugins import Setting

# The class-weighted checkpoint. The unweighted one answered `form` whenever it
# was unsure -- 617 forms against 128 resumes in training -- and scored 0.821 on
# fax against this one's 0.893.
DEFAULT_MODEL = os.environ.get("DI_LAYOUT_MODEL", "models/layout-balanced")


@register("layout")
class Layout:
    """LayoutLM over words, word boxes and the page image."""

    SETTINGS = (
        Setting("model", str, default=DEFAULT_MODEL,
                help="directory holding the fine-tuned checkpoint"),
        Setting("device", str, default="auto", help="auto | cuda | cpu"),
        Setting("abstain_below", float, default=0.0,
                help="say nothing when the top probability is under this; 0 disables"),
    )

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto",
                 abstain_below: float = 0.0, **_):
        self.model_dir = model
        self.device_choice = device
        self.abstain_below = abstain_below
        self._model = None
        self._processor = None
        self._device = None

    def describe(self) -> str:
        return f"layout - {self.model_dir}"

    def _load(self):
        if self._model is not None:
            return
        if not os.path.isdir(self.model_dir):
            raise SystemExit(
                f"no fine-tuned checkpoint at {self.model_dir!r}.\n"
                f"  Train one first:  python tools/train-layout-classifier.py")
        import torch
        from transformers import AutoProcessor, AutoModelForSequenceClassification

        if self.device_choice == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device_choice
        self._processor = AutoProcessor.from_pretrained(self.model_dir, apply_ocr=False)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir).to(self._device).eval()

    def classify(self, text: str = "", document=None, path: str = "",
                 **_) -> Classification:
        """Classify from layout. `text` is ignored; this stage reads boxes and pixels.

        The signature keeps `text` first so this plugin is interchangeable with the
        text classifiers at the call site, but a caller that passes only a string is
        giving this one nothing to work with, and it says so rather than guessing.
        """
        started = time.time()
        if document is None or not path:
            raise SystemExit(
                "the layout classifier needs the document and its path, not just "
                "text; the caller passed only a string")

        import torch
        from classify.features import features

        self._load()
        words, boxes, image = features(path, document.words)
        if not words:
            # Nothing legible and nothing positioned. Abstaining is the honest answer
            # and routes the page to a person, which is where it belongs.
            return Classification(doc_type="", engine=f"layout:{self.model_dir}",
                                  evidence="no words with usable boxes",
                                  seconds=time.time() - started)

        encoded = self._processor([image], [words], boxes=[boxes], truncation=True,
                                  padding=True, max_length=512, return_tensors="pt")
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits.float()
        probability = torch.softmax(logits, dim=-1)[0]
        order = probability.argsort(descending=True)
        labels = self._model.config.id2label
        best = labels[order[0].item()]
        confidence = probability[order[0]].item()
        # The head is trained on `form:w9`, not `form`: the variant is what selects
        # the field set, and returning only the type would leave that to the corpus.
        doc_type, variant = split_label(best)

        withheld = ""
        if self.abstain_below and confidence < self.abstain_below:
            # Kept, not dropped: see the note in the DiT plugin. A declined document
            # whose suppressed answer was right and one whose answer was wrong are the
            # difference between a floor set too high and a floor doing its job.
            withheld, doc_type, variant = best, "", ""
        return Classification(
            doc_type=doc_type, variant=variant, withheld=withheld,
            confidence=round(confidence, 4),
            margin=round(confidence - probability[order[1]].item(), 4),
            runner_up=split_label(labels[order[1].item()])[0],
            evidence=f"{len(words)} words with boxes on page one",
            engine=f"layout:{os.path.basename(self.model_dir)}",
            seconds=time.time() - started)
