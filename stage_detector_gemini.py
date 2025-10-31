# stage_detector_gemini.py
# Hybrid rule + Gemini Embeddings stage detection with temporal smoothing and weighted aggregation.
# Requires: pip install google-generativeai numpy scikit-learn
from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import google.generativeai as genai
except Exception:
    genai = None  # type: ignore

Stage = str
STAGES: Tuple[Stage, ...] = ("clarify", "plan", "execute", "verify", "reflect")

# -----------------------------
# Config + Prototypes
# -----------------------------
@dataclass
class DetectorConfig:
    lambda_fusion: float = 0.5     # weight for rules in fusion
    beta_smooth: float = 0.7       # temporal smoothing
    tau: float = 10.0              # temperature for softmax on cosine sims
    # weights for aggregation (must sum to 1.0)
    gamma_time: float = 0.4
    gamma_len: float = 0.4
    gamma_event: float = 0.2
    # dwell time capping
    dwell_cap_seconds: float = 300.0
    entropy_thresh: float = 1.4
    # Gemini config
    gemini_model: str = "text-embedding-004"
    google_api_key_env: str = "GOOGLE_API_KEY"

DEFAULT_PROTOTYPES: Dict[Stage, str] = {
    "clarify": "Clarify: asking for understanding; questions; definitions; uncertainty; what does it mean; can you explain",
    "plan": "Plan: outlining an approach; plan; strategy; steps; I will try; first then finally; we should attempt",
    "execute": "Execute: implementing; running code; logs; error message; compile; run; output shows; debugging traceback",
    "verify": "Verify: testing correctness; I checked; compared; it passed or failed; metrics; accuracy; unit tests results",
    "reflect": "Reflect: summarizing learning; I learned; I realized; next time I will; key takeaway; in hindsight",
}

# -----------------------------
# Utilities
# -----------------------------
def softmax(arr: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    z = (x - x.max()) * tau
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return e / s

def normalize_probs(v: Dict[Stage, float]) -> Dict[Stage, float]:
    total = sum(max(0.0, v.get(k, 0.0)) for k in STAGES)
    if total <= 0:
        return {k: 1.0/len(STAGES) for k in STAGES}
    return {k: max(0.0, v.get(k, 0.0)) / total for k in STAGES}

def entropy(p: Dict[Stage, float]) -> float:
    eps = 1e-12
    return -sum(pi * math.log(pi + eps) for pi in p.values())

def token_len(text: str) -> int:
    return max(1, len(re.findall(r"\w+", text or "")))

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# -----------------------------
# Rules (regex-based scoring)
# -----------------------------
RULES: Dict[Stage, List[re.Pattern]] = {
    "clarify": [
        re.compile(r"^\s*(what|why|how|which|where)\b", re.I),
        re.compile(r"\b(don't|do not|cannot|can't)\s+(understand|get)\b", re.I),
        re.compile(r"\bdefinition of\b", re.I),
        re.compile(r"\b(explain|clarify)\b", re.I),
        re.compile(r"\?\s*$"),
    ],
    "plan": [
        re.compile(r"\b(plan|approach|strategy|steps?)\b", re.I),
        re.compile(r"\b(i|we)\s+(will|shall|gonna|going to|intend to|aim to)\b", re.I),
        re.compile(r"\b(first|then|finally|next)\b", re.I),
        re.compile(r"\btry(ing)? to\b", re.I),
    ],
    "execute": [
        re.compile(r"```|`{3,}|<code>|</code>", re.I),
        re.compile(r"\b(run|ran|execute|executed|compile|compiled)\b", re.I),
        re.compile(r"\b(error|traceback|exception|stack trace|output)\b", re.I),
        re.compile(r"\bprint(ed)?\b", re.I),
    ],
    "verify": [
        re.compile(r"\b(check|checked|verify|verified|validate|validated)\b", re.I),
        re.compile(r"\b(compared|compare|diff)\b", re.I),
        re.compile(r"\b(passed|failed|success|correct|incorrect)\b", re.I),
        re.compile(r"\b(accuracy|r2|rmse|precision|recall|f1|auc)\b", re.I),
        re.compile(r"\b(unit\s*test|assert)\b", re.I),
    ],
    "reflect": [
        re.compile(r"\b(i|we)\s+(learned|realized|noticed|understood)\b", re.I),
        re.compile(r"\b(next time|in hindsight|takeaway|lesson)\b", re.I),
        re.compile(r"\b(should have|would have|could have)\b", re.I),
        re.compile(r"\b(retrospective|reflection)\b", re.I),
    ],
}

def rules_to_prior(text: str) -> Dict[Stage, float]:
    raw: Dict[Stage, float] = {k: 0.0 for k in STAGES}
    t = text or ""
    for stage, patterns in RULES.items():
        score = 0.0
        for pat in patterns:
            if pat.search(t):
                score += 1.0
        if score > 0:
            score = 1.0 + math.log1p(score)
        raw[stage] = score
    return normalize_probs(raw)

# -----------------------------
# Gemini Embedding Encoder
# -----------------------------
@dataclass
class GeminiEncoder:
    prototypes: Dict[Stage, str] = field(default_factory=lambda: DEFAULT_PROTOTYPES.copy())
    model_name: str = "text-embedding-004"
    api_key_env: str = "GOOGLE_API_KEY"

    _proto_vecs: Optional[Dict[Stage, np.ndarray]] = field(default=None, init=False, repr=False)
    _cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _configured: bool = field(default=False, init=False, repr=False)

    def _ensure_client(self):
        if genai is None:
            raise RuntimeError("google-generativeai is not installed. Run: pip install google-generativeai")
        if not self._configured:
            key = os.environ.get(self.api_key_env, "")
            if not key:
                raise RuntimeError(f"Missing Google API key in environment variable {self.api_key_env}.")
            genai.configure(api_key=key)
            self._configured = True

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        self._ensure_client()
        resp = genai.embed_content(model=self.model_name, content=text)
        vec = np.array(resp["embedding"], dtype=float)
        self._cache[text] = vec
        return vec

    def fit(self):
        # Pre-compute prototype embeddings
        _ = self.prototype_vectors
        return self

    @property
    def prototype_vectors(self) -> Dict[Stage, np.ndarray]:
        if self._proto_vecs is None:
            self._proto_vecs = {stage: self.embed(self.prototypes[stage]) for stage in STAGES}
        return self._proto_vecs

    def similarities(self, text: str) -> Dict[Stage, float]:
        q = self.embed(text or "")
        sims = {stage: cosine(q, vec) for stage, vec in self.prototype_vectors.items()}
        return sims

    def q_distribution(self, text: str, tau: float) -> Dict[Stage, float]:
        sims = self.similarities(text)
        arr = np.array([sims[s] for s in STAGES], dtype=float)
        probs = softmax(arr, tau=tau)
        return {stage: float(probs[i]) for i, stage in enumerate(STAGES)}

# -----------------------------
# Core detector
# -----------------------------
@dataclass
class Message:
    text: str
    timestamp: Optional[float] = None
    has_event: bool = False

@dataclass
class MessageResult:
    p_rule: Dict[Stage, float]
    p_sem: Dict[Stage, float]
    p_fused: Dict[Stage, float]
    p_smoothed: Dict[Stage, float]
    weight: float
    ambiguous: bool

@dataclass
class SessionResult:
    per_message: List[MessageResult]
    pi: Dict[Stage, float]
    weights_sum: float

class StageDetectorGemini:
    def __init__(self, cfg: Optional[DetectorConfig] = None, encoder: Optional[GeminiEncoder] = None):
        self.cfg = cfg or DetectorConfig()
        self.encoder = encoder or GeminiEncoder(model_name=self.cfg.gemini_model, api_key_env=self.cfg.google_api_key_env).fit()

    def fuse(self, r: Dict[Stage, float], q: Dict[Stage, float], lam: float) -> Dict[Stage, float]:
        unnorm = {}
        for s in STAGES:
            unnorm[s] = (max(r[s], 1e-12) ** lam) * (max(q[s], 1e-12) ** (1 - lam))
        return normalize_probs(unnorm)

    def smooth(self, prev: Dict[Stage, float], cur_hat: Dict[Stage, float]) -> Dict[Stage, float]:
        beta = self.cfg.beta_smooth
        out = {s: beta * cur_hat[s] + (1 - beta) * prev[s] for s in STAGES}
        return normalize_probs(out)

    def compute_weights(self, msgs: Sequence[Message]) -> List[float]:
        cfg = self.cfg
        n = len(msgs)
        deltas: List[float] = []
        for i, m in enumerate(msgs):
            if i == 0 or m.timestamp is None or msgs[i-1].timestamp is None:
                deltas.append(1.0)
            else:
                d = max(0.0, m.timestamp - msgs[i-1].timestamp)
                d = min(d, cfg.dwell_cap_seconds)
                deltas.append(d if d > 0 else 1.0)

        lens = [float(token_len(m.text)) for m in msgs]
        events = [1.0 if m.has_event else 0.0 for m in msgs]

        def norm(arr: List[float]) -> List[float]:
            a = np.array(arr, dtype=float)
            s = a.sum()
            if s <= 0:
                return [1.0 for _ in arr]
            return list(a / s)

        nd = norm(deltas)
        nl = norm(lens)
        ne = norm(events) if sum(events) > 0 else [0.0 for _ in events]

        w = [
            cfg.gamma_time * nd[i] + cfg.gamma_len * nl[i] + cfg.gamma_event * ne[i]
            for i in range(n)
        ]
        return [max(1e-9, wi) for wi in w]

    def process(self, msgs: Sequence[Message]) -> SessionResult:
        if not msgs:
            empty = {s: 0.0 for s in STAGES}
            return SessionResult([], empty, 0.0)

        weights = self.compute_weights(msgs)
        prev = {s: 1.0 / len(STAGES) for s in STAGES}
        results: List[MessageResult] = []

        for m, w in zip(msgs, weights):
            r = rules_to_prior(m.text)
            q = self.encoder.q_distribution(m.text, tau=self.cfg.tau)

            L = token_len(m.text)
            lam = min(0.8, max(0.2, self.cfg.lambda_fusion + 0.2 * (50 - L) / 50.0))
            fused = self.fuse(r, q, lam=lam)
            if L < 10:
                fused = normalize_probs({k: 0.6 * r[k] + 0.4 * fused[k] for k in STAGES})

            smoothed = self.smooth(prev, fused)
            prev = smoothed
            amb = entropy(fused) >= self.cfg.entropy_thresh

            results.append(
                MessageResult(
                    p_rule=r, p_sem=q, p_fused=fused, p_smoothed=smoothed, weight=w, ambiguous=amb
                )
            )

        Z = sum(weights) or 1.0
        pi = {s: 0.0 for s in STAGES}
        for res in results:
            for s in STAGES:
                pi[s] += res.p_smoothed[s] * res.weight
        for s in STAGES:
            pi[s] /= Z

        return SessionResult(per_message=results, pi=pi, weights_sum=Z)

# Demo (CLI-friendly)
if __name__ == "__main__":
    cfg = DetectorConfig()
    det = StageDetectorGemini(cfg=cfg)
    msgs = [
        Message("What does gradient clipping actually do? I don't understand exploding gradients."),
        Message("I'll start by limiting the norm and then compare training curves."),
        Message("I ran the training for 2 epochs; the log shows loss=1.2 then 0.9"),
        Message("I checked validation accuracy; it improved from 60% to 68% compared to baseline."),
        Message("I learned that clipping stabilizes updates; next time I'll tune the threshold earlier."),
    ]
    res = det.process(msgs)
    print("Session percentages (pi):")
    for k, v in res.pi.items():
        print(f"  {k:8s}: {v:.3f}")
    print("\nPer-message dominant stages:")
    for i, r in enumerate(res.per_message, 1):
        dom = max(r.p_smoothed, key=r.p_smoothed.get)
        print(f"  t={i}: {dom} (ambiguous={r.ambiguous})")
