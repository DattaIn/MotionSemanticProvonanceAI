"""
MotionSemanticProvenanceAI – Proof of Concept
==============================================
Meta Chain-of-Thought Guided Physics-Semantic Provenance Verification
for Detecting AI-Generated Media

Paper: "Meta Chain-of-Thought Guided Physics-Semantic Provenance Verification
        for Detecting AI-Generated Media" — Indranill Datta, 2026 (Pre-Print)

POC Scope
---------
* Synthetic video databases mimicking the 11-dataset benchmark from Table III
* Full 9-stage MotionSemanticProvenanceAI pipeline (Algorithm 1)
* Meta-CoT reasoning trace generation
* Cryptographic provenance record (SHA-256)
* Evaluation metrics from Table IV
"""

import hashlib
import json
import time
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC DATABASE GENERATOR
#     Mimics the 11 benchmark datasets in Table III
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticDatabase:
    """
    Generates synthetic optical-flow magnitude sequences f(t) for each dataset.

    Real/Authentic datasets → smooth Newtonian trajectories
      UCF101, Kinetics-700, DAVIS, ImageNet-1K, COCO 2017

    AI-Generated datasets   → elevated jerk variance + spectral flatness
      Stable Diffusion v2, Midjourney v6, DALL-E 3,
      Sora (sample clips), Runway Gen-2, Pika Labs 1.0
    """

    DATASETS = {
        # name            : (category,  label,  motion_noise, jerk_scale, sf_mean)
        "UCF101"          : ("Video",   "real",  0.05,  0.8,   0.38),
        "Kinetics-700"    : ("Video",   "real",  0.06,  0.9,   0.41),
        "DAVIS"           : ("Video",   "real",  0.04,  0.7,   0.36),
        "ImageNet-1K"     : ("Image",   "real",  0.03,  0.5,   0.43),
        "COCO-2017"       : ("Image",   "real",  0.04,  0.6,   0.44),
        "StableDiffusion" : ("Image",   "ai",    0.18,  4.2,   0.79),
        "Midjourney-v6"   : ("Image",   "ai",    0.20,  4.8,   0.82),
        "DALL-E-3"        : ("Image",   "ai",    0.16,  4.0,   0.77),
        "Sora"            : ("Video",   "ai",    0.25,  5.5,   0.85),
        "RunwayGen2"      : ("Video",   "ai",    0.22,  5.0,   0.83),
        "PikaLabs"        : ("Video",   "ai",    0.21,  4.9,   0.81),
    }

    REAL_SENSOR_SF_BASELINE = 0.43   # μ from Table III / Sec 3.2
    REAL_SENSOR_SF_STD      = 0.09
    JERK_NATURAL_THRESHOLD  = 4.2    # σ²(j) upper bound for real motion
    ENTROPY_THRESHOLD       = 0.15   # Δ motion entropy spike

    def __init__(self, n_frames: int = 90, seed: int = 42):
        self.n_frames = n_frames
        self.rng = np.random.default_rng(seed)

    def _generate_flow_sequence(self, motion_noise: float, jerk_scale: float) -> np.ndarray:
        """Synthesise f(t): smooth base + noise shaped by authenticity category."""
        t = np.linspace(0, 2 * math.pi, self.n_frames)
        # Natural sinusoidal motion base
        base = 2.0 + 1.5 * np.sin(t) + 0.5 * np.sin(3 * t)
        # Add noise whose 3rd derivative (jerk) grows with jerk_scale
        noise = self.rng.normal(0, motion_noise, self.n_frames)
        for _ in range(int(jerk_scale)):          # Introduce abrupt jumps
            idx = self.rng.integers(5, self.n_frames - 5)
            noise[idx] += self.rng.normal(0, motion_noise * jerk_scale)
        return np.maximum(base + noise, 0.0)

    def _generate_spectral_flatness(self, sf_mean: float) -> float:
        """Sample a single spectral flatness value."""
        return float(np.clip(self.rng.normal(sf_mean, 0.04), 0.0, 1.0))

    def generate_sample(self, dataset_name: str) -> Dict:
        """Return one synthetic media sample with ground-truth label."""
        cat, label, mn, js, sf_m = self.DATASETS[dataset_name]
        flow = self._generate_flow_sequence(mn, js)
        sf   = self._generate_spectral_flatness(sf_m)
        return {
            "dataset"  : dataset_name,
            "category" : cat,
            "label"    : label,           # "real" | "ai"
            "flow_seq" : flow.tolist(),
            "sf"       : sf,
        }

    def generate_dataset(self, samples_per_dataset: int = 50) -> List[Dict]:
        """Generate balanced benchmark corpus."""
        corpus = []
        for ds_name in self.DATASETS:
            for _ in range(samples_per_dataset):
                corpus.append(self.generate_sample(ds_name))
        return corpus


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PHYSICS METRICS (Section III.1 – III.2)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicsMetrics:
    jerk_variance    : float
    motion_entropy   : float
    spectral_flatness: float
    spectral_anomaly : float  # |SF − SF_baseline|
    accel_variance   : float


def compute_physics_metrics(flow_seq: List[float], sf: float) -> PhysicsMetrics:
    f  = np.array(flow_seq)
    a  = np.diff(f)                          # acceleration aₜ = fₜ − fₜ₋₁
    j  = np.diff(a)                          # jerk        jₜ = aₜ − aₜ₋₁

    jerk_var  = float(np.var(j))
    accel_var = float(np.var(a))

    # Motion entropy over flow magnitude distribution
    hist, _ = np.histogram(f, bins=20, density=True)
    hist    = hist[hist > 0]
    entropy = float(stats.entropy(hist))

    delta_sf = abs(sf - SyntheticDatabase.REAL_SENSOR_SF_BASELINE)

    return PhysicsMetrics(
        jerk_variance     = jerk_var,
        motion_entropy    = entropy,
        spectral_flatness = sf,
        spectral_anomaly  = delta_sf,
        accel_variance    = accel_var,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3.  META-COT REASONING ENGINE (Section III.3)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningTrace:
    observation   : str
    hypothesis    : str
    verification  : str
    conclusion    : str
    ai_probability: float
    verified      : bool


def _find_dominant_anomaly(pm: PhysicsMetrics, db: SyntheticDatabase) -> Tuple[str, str]:
    """Return (observation_text, hypothesis_text) for the worst anomaly found."""
    issues = []
    if pm.jerk_variance > db.JERK_NATURAL_THRESHOLD:
        issues.append((pm.jerk_variance, "jerk",
                       f"jerk variance σ²(j)={pm.jerk_variance:.2f} "
                       f"(natural threshold: {db.JERK_NATURAL_THRESHOLD})",
                       "Frame interpolation / generative upsampling artefact"))
    if pm.spectral_anomaly > 0.25:
        issues.append((pm.spectral_anomaly, "spectral",
                       f"spectral flatness SF={pm.spectral_flatness:.3f} deviates "
                       f"from real-sensor baseline "
                       f"SF={db.REAL_SENSOR_SF_BASELINE}±{db.REAL_SENSOR_SF_STD}",
                       "Over-smoothing from score-matching / diffusion objective"))
    if pm.motion_entropy > 2.8:
        issues.append((pm.motion_entropy, "entropy",
                       f"motion entropy H(f)={pm.motion_entropy:.3f} exceeds natural "
                       f"bound (2.8)",
                       "Temporal distribution shift at synthetic scene boundary"))
    if not issues:
        return ("No significant physics anomaly detected",
                "Media likely authentic — all metrics within natural bounds")
    issues.sort(key=lambda x: -x[0])
    return issues[0][2], issues[0][3]


def meta_cot_reason(pm: PhysicsMetrics, db: SyntheticDatabase,
                    prior_trace: Optional[ReasoningTrace] = None) -> ReasoningTrace:
    """
    Algorithm 1, lines 11-14: generate (and optionally re-run) Meta-CoT trace.
    """
    obs_text, hyp_text = _find_dominant_anomaly(pm, db)

    # Verification: check all three physics constraints
    jerk_fail    = pm.jerk_variance    > db.JERK_NATURAL_THRESHOLD
    sf_fail      = pm.spectral_anomaly > 0.25
    entropy_fail = pm.motion_entropy   > 2.8

    n_fails    = sum([jerk_fail, sf_fail, entropy_fail])
    verified   = (n_fails >= 1)
    ver_parts  = []
    if jerk_fail:
        ver_parts.append(f"σ²(j)={pm.jerk_variance:.2f} EXCEEDS threshold {db.JERK_NATURAL_THRESHOLD}")
    if sf_fail:
        ver_parts.append(f"ΔSF={pm.spectral_anomaly:.3f} EXCEEDS anomaly limit 0.25")
    if entropy_fail:
        ver_parts.append(f"H(f)={pm.motion_entropy:.3f} EXCEEDS entropy limit 2.8")
    if not ver_parts:
        ver_parts.append("All physics constraints SATISFIED — consistent with real media")

    ver_text = "; ".join(ver_parts)

    # Logistic scoring from physics deviations
    score = 0.0
    score += min(pm.jerk_variance / (db.JERK_NATURAL_THRESHOLD * 3), 0.45)
    score += min(pm.spectral_anomaly / 0.50, 0.35)
    score += min(pm.motion_entropy   / 6.0,  0.20)
    if prior_trace:
        score = 0.6 * score + 0.4 * prior_trace.ai_probability

    ai_prob = float(np.clip(score, 0.0, 1.0))
    verdict = "AI-GENERATED" if ai_prob >= 0.5 else "AUTHENTIC"
    conclusion = (f"Tamper probability P(AI)={ai_prob:.3f} — media classified as {verdict}. "
                  f"({n_fails}/3 physics constraints violated)")

    return ReasoningTrace(
        observation    = obs_text,
        hypothesis     = hyp_text,
        verification   = ver_text,
        conclusion     = conclusion,
        ai_probability = ai_prob,
        verified       = verified,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CRYPTOGRAPHIC PROVENANCE LAYER (Section III.4)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProvenanceRecord:
    h1_media_hash  : str
    h2_metrics_hash: str
    h3_trace_hash  : str
    timestamp_utc  : str
    composite_id   : str


def compute_provenance(sample: Dict, pm: PhysicsMetrics, trace: ReasoningTrace) -> ProvenanceRecord:
    h1 = hashlib.sha256(json.dumps(sample,      sort_keys=True).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps(asdict(pm),  sort_keys=True).encode()).hexdigest()
    h3 = hashlib.sha256(json.dumps(asdict(trace), sort_keys=True).encode()).hexdigest()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    composite = hashlib.sha256((h1 + h2 + h3 + ts).encode()).hexdigest()[:16].upper()
    return ProvenanceRecord(
        h1_media_hash   = h1[:16] + "…",
        h2_metrics_hash = h2[:16] + "…",
        h3_trace_hash   = h3[:16] + "…",
        timestamp_utc   = ts,
        composite_id    = composite,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  FULL 9-STAGE PIPELINE (Algorithm 1)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    dataset       : str
    category      : str
    true_label    : str
    predicted_label: str
    ai_probability: float
    physics       : PhysicsMetrics
    trace         : ReasoningTrace
    provenance    : ProvenanceRecord


def run_pipeline(sample: Dict, db: SyntheticDatabase) -> VerificationResult:
    """
    Algorithm 1 — 9-stage MotionSemanticProvenanceAI pipeline.
    Stage ①  Input Video          — sample already loaded
    Stage ②  Optical Flow         — pre-computed in synthetic DB (flow_seq)
    Stage ③④⑤ Physics metrics     — compute_physics_metrics()
    Stage ⑥  Spectral Flatness    — embedded in PhysicsMetrics
    Stage ⑦  Meta-CoT Reasoning   — meta_cot_reason()
    Stage ⑧  Provenance Hashing   — compute_provenance()
    Stage ⑨  Authenticity Score   — trace.ai_probability
    """
    # Stages ②–⑥
    pm = compute_physics_metrics(sample["flow_seq"], sample["sf"])

    # Stage ⑦  (with possible second-pass if verification fails first time)
    trace = meta_cot_reason(pm, db)
    if not trace.verified and trace.ai_probability > 0.4:
        trace = meta_cot_reason(pm, db, prior_trace=trace)   # line 12-14

    # Stage ⑧
    prov = compute_provenance(sample, pm, trace)

    # Stage ⑨
    predicted = "ai" if trace.ai_probability >= 0.5 else "real"

    return VerificationResult(
        dataset        = sample["dataset"],
        category       = sample["category"],
        true_label     = sample["label"],
        predicted_label= predicted,
        ai_probability = trace.ai_probability,
        physics        = pm,
        trace          = trace,
        provenance     = prov,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6.  EVALUATION (Table IV metrics)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(results: List[VerificationResult]) -> Dict:
    y_true = [1 if r.true_label == "ai" else 0 for r in results]
    y_pred = [1 if r.predicted_label == "ai" else 0 for r in results]

    acc     = accuracy_score(y_true, y_pred)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr     = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Reasoning Consistency Score (RCS) — fraction of traces with all 3 criteria met
    def _rcs(trace: ReasoningTrace) -> bool:
        return (bool(trace.observation) and
                bool(trace.hypothesis)  and
                bool(trace.verification) and
                bool(trace.conclusion))
    rcs = sum(_rcs(r.trace) for r in results) / len(results)

    # Physics divergence metrics (Table IV)
    jerk_vars = [r.physics.jerk_variance    for r in results]
    sf_deltas = [r.physics.spectral_anomaly for r in results]
    entropies = [r.physics.motion_entropy   for r in results]

    real_jv  = [jerk_vars[i] for i, r in enumerate(results) if r.true_label == "real"]
    ai_jv    = [jerk_vars[i] for i, r in enumerate(results) if r.true_label == "ai"]
    jerk_div = abs(np.mean(ai_jv) - np.mean(real_jv)) if (real_jv and ai_jv) else 0.0

    real_sf  = [sf_deltas[i] for i, r in enumerate(results) if r.true_label == "real"]
    ai_sf    = [sf_deltas[i] for i, r in enumerate(results) if r.true_label == "ai"]
    sf_div   = abs(np.mean(ai_sf) - np.mean(real_sf)) if (real_sf and ai_sf) else 0.0

    return {
        "accuracy"           : round(acc, 4),
        "false_positive_rate": round(fpr, 4),
        "reasoning_consistency_score": round(rcs, 4),
        "jerk_variance_divergence"   : round(jerk_div, 4),
        "spectral_divergence"        : round(sf_div, 4),
        "mean_jerk_variance_real"    : round(float(np.mean(real_jv)), 4),
        "mean_jerk_variance_ai"      : round(float(np.mean(ai_jv)), 4),
        "total_samples"              : len(results),
        "true_positives"             : int(tp),
        "true_negatives"             : int(tn),
        "false_positives"            : int(fp),
        "false_negatives"            : int(fn),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  RUNNER  —  entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MotionSemanticProvenanceAI — Proof of Concept")
    print("  Physics-Semantic Provenance Verification for AI Media Detection")
    print("=" * 70)

    db     = SyntheticDatabase(n_frames=90, seed=2026)
    corpus = db.generate_dataset(samples_per_dataset=50)
    print(f"\n[DB] Generated {len(corpus)} samples across {len(db.DATASETS)} datasets")

    # ── Run full pipeline on every sample ────────────────────────────────────
    results: List[VerificationResult] = []
    for sample in corpus:
        results.append(run_pipeline(sample, db))

    # ── Show one detailed trace per dataset ──────────────────────────────────
    seen = set()
    print("\n" + "─" * 70)
    print("  SAMPLE VERIFICATION TRACES (one per dataset)")
    print("─" * 70)
    for r in results:
        if r.dataset not in seen:
            seen.add(r.dataset)
            print(f"\n📂 Dataset      : {r.dataset} ({r.category}) | True={r.true_label.upper()}")
            print(f"   Observation  : {r.trace.observation}")
            print(f"   Hypothesis   : {r.trace.hypothesis}")
            print(f"   Verification : {r.trace.verification}")
            print(f"   Conclusion   : {r.trace.conclusion}")
            print(f"   Predicted    : {r.predicted_label.upper()} ({'✓ CORRECT' if r.predicted_label == r.true_label else '✗ WRONG'})")
            print(f"   Provenance   : ID={r.provenance.composite_id}  ts={r.provenance.timestamp_utc}")

    # ── Evaluation report ────────────────────────────────────────────────────
    metrics = evaluate(results)
    print("\n" + "─" * 70)
    print("  EVALUATION RESULTS  (Table IV Metrics)")
    print("─" * 70)
    print(f"  Accuracy                     : {metrics['accuracy']*100:.2f}%")
    print(f"  False Positive Rate          : {metrics['false_positive_rate']*100:.2f}%")
    print(f"  Reasoning Consistency Score  : {metrics['reasoning_consistency_score']*100:.1f}%")
    print(f"  Jerk Variance Divergence     : {metrics['jerk_variance_divergence']:.4f}")
    print(f"  Spectral Divergence          : {metrics['spectral_divergence']:.4f}")
    print(f"  Mean Jerk Var (Real)         : {metrics['mean_jerk_variance_real']:.4f}")
    print(f"  Mean Jerk Var (AI)           : {metrics['mean_jerk_variance_ai']:.4f}")
    print(f"  TP/TN/FP/FN                  : {metrics['true_positives']}/{metrics['true_negatives']}/{metrics['false_positives']}/{metrics['false_negatives']}")
    print(f"  Total samples                : {metrics['total_samples']}")

    # ── Per-dataset breakdown ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PER-DATASET ACCURACY")
    print("─" * 70)
    from collections import defaultdict
    ds_correct = defaultdict(int)
    ds_total   = defaultdict(int)
    for r in results:
        ds_total[r.dataset]   += 1
        ds_correct[r.dataset] += (1 if r.predicted_label == r.true_label else 0)
    for ds in db.DATASETS:
        acc = ds_correct[ds] / ds_total[ds] * 100
        lbl = db.DATASETS[ds][1].upper()
        print(f"  {ds:<20} [{lbl:4s}]  {acc:6.1f}%  ({ds_correct[ds]}/{ds_total[ds]})")

    print("\n[POC] Complete.\n")
    return metrics, results


if __name__ == "__main__":
    main()
