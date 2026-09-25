"""B3 classifier adapters — a local small model behind the hook.

The resolver's Path B3 (see :mod:`llmrouter.signals`) is where a router pays
a little latency once, on turn 1, to amortize a better workload class across a
whole session. It was always a protocol with no bundled implementation; this
module ships one that fits the economics exactly.

``LayaWorkloadClassifier`` talks to a local
[Laya](https://github.com/NandhaKishorM/laya) server (FastAPI ``POST
/predict``) and asks one ``choice`` question — *"which workload is this?"* —
over the six workload classes. Laya answers in ~33 ms with a single
non-autoregressive forward pass: no generation, so there is nothing to parse
and nothing to hallucinate, and calibration is trained against proper scoring
rules, which is precisely what a confidence we can use should mean.

The adapter is stdlib-only (``urllib``) so the zero-dependency rule holds —
Laya itself runs as a separate process (its checkpoint, torch, CUDA all stay
over there). Any failure returns ``None``: the classifier is an UPGRADE on the
abstain path, never a dependency. If it is down, routing falls back to the
fingerprint registry and the default exactly as before.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Mapping

from .types import RoutingRequest, WorkloadClass, WorkloadSource
from .signals import WorkloadSignal

DEFAULT_LAYA_URL = "http://127.0.0.1:8000/predict"

#: The one choice-question we ask. Criteria descriptions are what Laya's
#: zero-shot scoring reads — keep them about the REQUEST, not the topic.
WORKLOAD_CRITERIA: Mapping[str, str] = {
    WorkloadClass.AGENT.value: ("an autonomous agent step: tool/function calls, "
                                "multi-step execution over a work session"),
    WorkloadClass.CHAT.value: "a conversational question or chit-chat reply",
    WorkloadClass.RAG.value: ("answering from retrieved or pasted context that "
                              "accompanies the question"),
    WorkloadClass.BATCH.value: ("a single-shot bulk transformation of many "
                                "similar items: classify, extract, summarize "
                                "one-off, no conversation"),
    WorkloadClass.CODE.value: ("writing, refactoring or explaining code; a "
                               "programming task"),
    WorkloadClass.VISION.value: "a request that involves images or visual input",
}


class LayaWorkloadClassifier:
    """Satisfies the ``WorkloadClassifier`` protocol via Laya's HTTP API.

    Wire it into any resolver construction::

        Router.with_ollama(
            classifier=LayaWorkloadClassifier("http://127.0.0.1:8000/predict"),
            ...
        )

    ``timeout_s`` bounds the turn-1 tax: the call happens once per new
    application prompt (the fingerprint registry learns the answer), so even
    the slow path is amortized across the session — but a hung server must
    never hang the router.
    """

    def __init__(self, endpoint: str = DEFAULT_LAYA_URL,
                 timeout_s: float = 2.0,
                 confidence_floor: float = 0.35,
                 question_key: str = "workload") -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        #: Below this the choice is noise — abstain and let the priors work.
        self.confidence_floor = confidence_floor
        #: The question name in the typed-decision payload. Codiv-hosted
        #: Verdict (rlcd-modernbert-151m / openJev) answers the same contract
        #: as Laya; any server conforming to it works with this adapter.
        self.question_key = question_key

    # ---- the protocol ---------------------------------------------------- #
    def classify(self, request: RoutingRequest) -> WorkloadSignal | None:
        text = request.first_user_text() or "\n".join(
            m.text for m in request.messages)
        if not text.strip():
            return None
        answer = self._ask(text)
        if answer is None:
            return None
        choice, confidence = answer
        try:
            workload = WorkloadClass(choice)
        except ValueError:
            return None  # a label outside our vocabulary is an abstention
        if confidence < self.confidence_floor:
            return None
        return WorkloadSignal(
            workload, WorkloadSource.CLASSIFIER, confidence,
            f"laya typed decision: {choice}",
        )

    # ---- transport -------------------------------------------------------- #
    def _ask(self, text: str) -> tuple[str, float] | None:
        payload = {
            "state": {"body": text},
            "questions": {
                self.question_key: {
                    "type": "choice",
                    "instructions": "Which single category best describes "
                                    "what this request needs?",
                    "criteria": dict(WORKLOAD_CRITERIA),
                }
            },
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read())
        except Exception:  # noqa: BLE001 — any failure is an abstention
            return None
        return self._parse(body)

    def _parse(self, body: Mapping[str, Any]) -> tuple[str, float] | None:
        ans = ((body.get("answers") or {}).get(self.question_key) or {})
        choice = ans.get("choice")
        if not isinstance(choice, str):
            return None
        # Laya variants expose the certainty differently; take the first real
        # number we find rather than pinning one schema.
        confidence: Any = None
        for key in ("probability", "confidence", "score"):
            if isinstance(ans.get(key), (int, float)):
                confidence = float(ans[key])
                break
        if confidence is None:
            scores = ans.get("scores") or {}
            if isinstance(scores, Mapping) and isinstance(scores.get(choice),
                                                          (int, float)):
                confidence = float(scores[choice])
        return choice, (0.8 if confidence is None else confidence)


# --------------------------------------------------------------------------- #
# In-process adapters: bring your own model
# --------------------------------------------------------------------------- #
LabelFn = Any  # Callable[[str], str | tuple[str, float] | None]


class CallableWorkloadClassifier:
    """Wrap any in-process text→label function as a B3 classifier.

    The escape hatch for models that prefer to run inside your process —
    SetFit, fastText, a scikit-learn head, a hash lookup, anything::

        from sentence_transformers import SentenceTransformer
        from setfit import SetFitModel
        model = SetFitModel.from_pretrained("me/workload-head")
        clf = CallableWorkloadClassifier(lambda text: model(text))

    The function may return:
      * a label string (confidence = ``default_confidence``)
      * a ``(label, confidence)`` tuple
      * ``None`` to abstain
    Exceptions are caught: a broken classifier must never break routing.
    """

    def __init__(self, fn: LabelFn, *, name: str = "callable",
                 default_confidence: float = 0.7,
                 confidence_floor: float = 0.35) -> None:
        self.fn = fn
        self.name = name
        self.default_confidence = default_confidence
        self.confidence_floor = confidence_floor

    def classify(self, request: RoutingRequest) -> WorkloadSignal | None:
        text = request.first_user_text() or "\n".join(
            m.text for m in request.messages)
        if not text.strip():
            return None
        try:
            out = self.fn(text)
        except Exception:  # noqa: BLE001 — abstention, never break routing
            return None
        confidence = self.default_confidence
        if isinstance(out, tuple):
            if len(out) != 2:
                return None
            choice, confidence = out[0], float(out[1])
        else:
            choice = out
        if not isinstance(choice, str):
            return None
        try:
            workload = WorkloadClass(choice)
        except ValueError:
            return None
        if confidence < self.confidence_floor:
            return None
        return WorkloadSignal(
            workload, WorkloadSource.CLASSIFIER, confidence,
            f"{self.name}: {choice}",
        )


class GLiClassWorkloadClassifier:
    """Zero-shot in-process classification via GLiClass (optional extra).

    GLiClass (Knowledgator) is the encoder family that backs the typed-decision
    models — a ModernBERT zero-shot head, single forward pass, ~10x faster than
    cross-encoders. Requires the user-side install ``pip install gliclass``
    (torch &co stay OUT of llmrouter's dependency graph; this adapter imports
    lazily and abstains cleanly when the package or model is missing).

    ::

        clf = GLiClassWorkloadClassifier("knowledgator/gliclass-small-v1.0")
    """

    def __init__(self, model_id: str = "knowledgator/gliclass-small-v1.0",
                 *, device: str | None = None,
                 confidence_floor: float = 0.35) -> None:
        self.model_id = model_id
        self.device = device
        self.confidence_floor = confidence_floor
        self._pipeline = None
        self._import_failed = False

    def _get_pipeline(self):
        if self._pipeline is not None or self._import_failed:
            return self._pipeline
        try:
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
        except ImportError:
            self._import_failed = True
            return None
        model = GLiClassModel.from_pretrained(self.model_id)
        tok = AutoTokenizer.from_pretrained(self.model_id)
        self._pipeline = ZeroShotClassificationPipeline(
            model, tok, classification_type="single-label",
            device=self.device,
        )
        return self._pipeline

    def classify(self, request: RoutingRequest) -> WorkloadSignal | None:
        pipe = self._get_pipeline()
        if pipe is None:
            return None
        text = request.first_user_text() or "\n".join(
            m.text for m in request.messages)
        if not text.strip():
            return None
        labels = list(WORKLOAD_CRITERIA)
        try:
            scores = pipe.run(text, labels)
        except Exception:  # noqa: BLE001 — abstention, never a crash
            return None
        # GLiClass returns per-text {label: score} dicts (or a list thereof).
        if isinstance(scores, list):
            scores = scores[0] if scores else {}
        if not isinstance(scores, Mapping):
            return None
        best, conf = None, -1.0
        for label, value in scores.items():
            if isinstance(value, (int, float)) and value > conf:
                best, conf = str(label), float(value)
        if best is None or best not in WorkloadClass.__members__.values():
            return None
        if conf < self.confidence_floor:
            return None
        return WorkloadSignal(
            WorkloadClass(best), WorkloadSource.CLASSIFIER, conf,
            f"gliclass {self.model_id.split('/')[-1]}: {best}",
        )


class ChainWorkloadClassifier:
    """Try classifiers in order; the first non-abstention wins.

    The 'whichever they want, whenever they want' answer: order is a policy,
    not a code change — cheap local head first, server model as backup, an LLM
    judge as last resort (or any permutation)::

        classifier=ChainWorkloadClassifier([
            GLiClassWorkloadClassifier(),                      # ~5 ms, in-proc
            LayaWorkloadClassifier("http://127.0.0.1:8000/predict"),  # ~33 ms
        ])
    """

    def __init__(self, classifiers: list) -> None:
        self.classifiers = list(classifiers)

    def classify(self, request: RoutingRequest) -> WorkloadSignal | None:
        for clf in self.classifiers:
            try:
                sig = clf.classify(request)
            except Exception:  # noqa: BLE001 — one bad link must not kill the chain
                continue
            if sig is not None:
                return sig
        return None


__all__ = [
    "LayaWorkloadClassifier", "DEFAULT_LAYA_URL", "WORKLOAD_CRITERIA",
    "CallableWorkloadClassifier", "GLiClassWorkloadClassifier",
    "ChainWorkloadClassifier",
]