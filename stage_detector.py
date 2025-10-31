# stage_detector.py
# Local TF-IDF fallback for the EDU-AI Cognitive Stage Detector.
# No API calls — all semantic similarity is computed via TF-IDF cosine.

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

STAGES = ["clarify", "plan", "execute", "verify", "reflect"]

# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------
@dataclass
class Message:
    text: str
    timestamp: Optional[float] = None
    has_event: bool = False


@dataclass
class DetectorConfig:
    lambda_fusion: float = 0.5
    beta_smooth: float = 0.7
    tau: float = 10.0
    gamma_time: float = 0.4
    gamma_len: float = 0.4
    gamma_event: float = 0.2


@dataclass
class MessageResult:
    p_rule: Dict[str, float]
    p_semantic: Dict[str, float]
    p_fused: Dict[str, float]
    p_smoothed: Dict[str, float]
    weight: float
    ambiguous: bool


@dataclass
class SessionResult:
    per_message: List[MessageResult]
    pi: Dict[str, float]


# -------------------------------------------------------------------
# StageDetector (local TF-IDF)
# -------------------------------------------------------------------
class StageDetector:
    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.stage_texts = {
            "clarify": "asking for understanding explanation meaning definition question",
            "plan": "outlining plan idea strategy next step maybe should approach",
            "execute": "running code implementation output error result test compute",
            "verify": "checking comparing validating confirming correct accuracy works",
            "reflect": "thinking reflecting learning realized next time improvement lesson",
        }
        self.stage_vectors = None
        self._fit_vectorizer()

    # -----------------------------
    def _fit_vectorizer(self):
        texts = list(self.stage_texts.values())
        self.vectorizer.fit(texts)
        X = self.vectorizer.transform(texts)
        self.stage_vectors = dict(zip(STAGES, X))

    # -----------------------------
    def _rule_based_scores(self, text: str) -> Dict[str, float]:
        text_low = text.lower()
        rule_patterns = {
            "clarify": [r"\bwhat\b", r"\bhow\b", r"\bwhy\b", r"\?\s*$", r"don't understand"],
            "plan": [r"\bplan\b", r"\bshould\b", r"\btry\b", r"\bmaybe\b", r"\bnext\b"],
            "execute": [r"\brun\b", r"\berror\b", r"\boutput\b", r"\bexecute\b", r"\bimplemented\b"],
            "verify": [r"\bcheck\b", r"\bcompare\b", r"\bvalidate\b", r"\bcorrect\b", r"\bworks?\b"],
            "reflect": [r"\blearned\b", r"\brealized\b", r"\bnext time\b", r"\bimprove\b", r"\bthink\b"],
        }
        scores = {}
        for stage, pats in rule_patterns.items():
            s = sum(bool(re.search(p, text_low)) for p in pats)
            scores[stage] = s
        total = sum(scores.values()) or 1
        for k in scores:
            scores[k] /= total
        return scores

    # -----------------------------
    def _semantic_scores(self, text: str) -> Dict[str, float]:
        v_msg = self.vectorizer.transform([text])
        sims = {s: cosine_similarity(v_msg, self.stage_vectors[s])[0, 0] for s in STAGES}
        sims = {k: math.exp(self.cfg.tau * v) for k, v in sims.items()}
        total = sum(sims.values()) or 1
        sims = {k: v / total for k, v in sims.items()}
        return sims

    # -----------------------------
    def _fuse(self, r: Dict[str, float], q: Dict[str, float]) -> Dict[str, float]:
        λ = self.cfg.lambda_fusion
        prod = {k: (r[k] ** λ) * (q[k] ** (1 - λ)) for k in STAGES}
        total = sum(prod.values()) or 1
        return {k: v / total for k, v in prod.items()}

    # -----------------------------
    def _smooth(self, p_prev: Optional[Dict[str, float]], p_hat: Dict[str, float]) -> Dict[str, float]:
        if p_prev is None:
            return p_hat
        β = self.cfg.beta_smooth
        return {k: β * p_hat[k] + (1 - β) * p_prev[k] for k in STAGES}

    # -----------------------------
    def _compute_weight(self, msg: Message, prev_msg: Optional[Message]) -> float:
        g_t, g_l, g_e = self.cfg.gamma_time, self.cfg.gamma_len, self.cfg.gamma_event
        time_w = 1.0
        if msg.timestamp and prev_msg and prev_msg.timestamp:
            time_w = max(0.1, min(5.0, msg.timestamp - prev_msg.timestamp))
        len_w = len(msg.text.split())
        event_w = 2.0 if msg.has_event else 1.0
        return g_t * time_w + g_l * len_w + g_e * event_w

    # -----------------------------
    def _entropy(self, p: Dict[str, float]) -> float:
        return -sum(p[k] * math.log(p[k] + 1e-12) for k in STAGES)

    # -----------------------------
    def process(self, messages: List[Message]) -> SessionResult:
        results = []
        p_prev = None
        for i, m in enumerate(messages):
            r = self._rule_based_scores(m.text)
            q = self._semantic_scores(m.text)
            p_hat = self._fuse(r, q)
            p_smooth = self._smooth(p_prev, p_hat)
            weight = self._compute_weight(m, messages[i-1] if i > 0 else None)
            ent = self._entropy(p_smooth)
            ambiguous = ent > 1.4  # arbitrary threshold
            results.append(MessageResult(r, q, p_hat, p_smooth, weight, ambiguous))
            p_prev = p_smooth

        # aggregate π
        weights = np.array([r.weight for r in results])
        weights = weights / (weights.sum() or 1)
        p_mat = np.array([[r.p_smoothed[k] for k in STAGES] for r in results])
        pi_vals = (weights[:, None] * p_mat).sum(axis=0)
        pi = dict(zip(STAGES, pi_vals / (pi_vals.sum() or 1)))

        return SessionResult(per_message=results, pi=pi)


# -------------------------------------------------------------------
# Quick manual test
# -------------------------------------------------------------------
if __name__ == "__main__":
    det = StageDetector()
    msgs = [
        Message("What does this function mean?"),
        Message("Maybe I should try a for loop."),
        Message("I ran the code and got an error message."),
        Message("I checked the output and it looks correct."),
        Message("I learned that list comprehension is faster."),
    ]
    res = det.process(msgs)
    print("Session π:", res.pi)
    for i, r in enumerate(res.per_message, 1):
        print(i, r.p_smoothed)