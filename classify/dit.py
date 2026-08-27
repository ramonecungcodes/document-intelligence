"""A classifier that only looks at the page.

DiT is a document image transformer: it reads the picture and nothing else -- no words,
no boxes, no OCR. On this corpus that turns out to be the whole job. Measured on the
same held-out faxes, with page designs the model had never seen in training:

    LLM, text only                          0.571
    LayoutLMv3, words + boxes + image       0.943
    DiT, image only                         0.958

Dropping the words made it better rather than worse, which is the opposite of the
intuition worth stating plainly: on a 170 dpi fax docTR finds 62% of the words, and a
model given ruined text alongside an intact page has been handed a second opinion that
is wrong exactly where the first one is struggling.

Two consequences follow from needing no text.

It runs before OCR. `NEEDS_TEXT = False` tells the runner not to normalize at all, so
classifying a corpus costs a page render instead of an OCR pass -- and the type it
returns is what selects the extraction schema anyway, which is upstream of the text.

And it degrades honestly. Confidence separates its right answers from its wrong ones:
on unseen-design faxes a 0.90 floor answers 88.7% of documents with no errors at all,
and every one of its mistakes arrives underneath that line. The threshold is set in
di.toml rather than defaulted here, so the number that governs the pipeline sits next
to the plugin that obeys it.

Licensing: check `microsoft/dit-base`'s own terms before shipping it. Do not assume it
inherits the MIT licence of the unilm repository it is published from.
"""
from __future__ import annotations

import os
import time

from classify.base import Classification, register
from core.plugins import Setting

DEFAULT_MODEL = os.environ.get("DI_DIT_MODEL", "models/dit-balanced")


@register("dit")
class DocumentImage:
    """Classify a document from a picture of its first page."""

    # The runner skips normalization entirely for this plugin. OCR is the expensive
    # stage and this one has no use for its output.
    NEEDS_TEXT = False

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
        floor = f" - abstains below {self.abstain_below}" if self.abstain_below else ""
        return f"dit - {self.model_dir}{floor}"

    def _load(self):
        if self._model is not None:
            return
        if not os.path.isdir(self.model_dir):
            raise SystemExit(
                f"no fine-tuned checkpoint at {self.model_dir!r}.\n"
                f"  Train one first:  python tools/train-layout-classifier.py "
                f"--arch image --checkpoint microsoft/dit-base --balance "
                f"--out {self.model_dir}")
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._device = ("cuda" if torch.cuda.is_available() else "cpu"
                        ) if self.device_choice == "auto" else self.device_choice
        self._processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self._model = AutoModelForImageClassification.from_pretrained(
            self.model_dir).to(self._device).eval()

    def classify(self, text: str = "", document=None, path: str = "",
                 **_) -> Classification:
        """Classify from the page image. `text` and `document` are ignored.

        The signature matches the text classifiers so the call site does not branch,
        but only `path` is read -- this plugin rasterises the document itself rather
        than depending on a stage that may not have run.
        """
        started = time.time()
        if not path:
            raise SystemExit("the dit classifier needs the document path, not text")

        import torch
        from classify.features import page_one

        self._load()
        _width, _height, image = page_one(path)
        encoded = self._processor([image], return_tensors="pt")
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits.float()
        probability = torch.softmax(logits, dim=-1)[0]
        order = probability.argsort(descending=True)
        labels = self._model.config.id2label
        best = labels[order[0].item()]
        confidence = probability[order[0]].item()

        if self.abstain_below and confidence < self.abstain_below:
            # Not a failure. Every one of this model's errors on unseen-design faxes
            # sat below 0.90, so declining here is the difference between a wrong
            # extraction schema and a document a person looks at.
            best = ""
        return Classification(
            doc_type=best,
            confidence=round(confidence, 4),
            runner_up=labels[order[1].item()],
            evidence="page image, no text read",
            engine=f"dit:{os.path.basename(self.model_dir)}",
            seconds=time.time() - started)
