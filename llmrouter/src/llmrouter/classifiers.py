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
                 confidence_floor: float = 0.35) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        #: Below this the choice is noise — abstain and let the priors work.
        self.confidence_floor = confidence_floor

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
                "workload": {
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

    @staticmethod
    def _parse(body: Mapping[str, Any]) -> tuple[str, float] | None:
        ans = ((body.get("answers") or {}).get("workload") or {})
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


__all__ = ["LayaWorkloadClassifier", "DEFAULT_LAYA_URL", "WORKLOAD_CRITERIA"]
