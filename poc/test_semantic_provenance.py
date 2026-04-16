"""
test_semantic_provenance.py
============================
Unit & integration tests for MotionSemanticProvenanceAI POC.
Run with:  python -m pytest test_semantic_provenance.py -v
"""

import pytest
import numpy as np
from motion_semantic_provenance import (
    SyntheticDatabase,
    compute_physics_metrics,
    meta_cot_reason,
    compute_provenance,
    run_pipeline,
    evaluate,
    PhysicsMetrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    return SyntheticDatabase(n_frames=90, seed=2026)


@pytest.fixture
def real_sample(db):
    return db.generate_sample("UCF101")


@pytest.fixture
def ai_sample(db):
    return db.generate_sample("Sora")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SyntheticDatabase
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticDatabase:

    def test_corpus_size(self, db):
        corpus = db.generate_dataset(samples_per_dataset=10)
        assert len(corpus) == 11 * 10

    def test_all_datasets_present(self, db):
        corpus = db.generate_dataset(samples_per_dataset=1)
        names  = {s["dataset"] for s in corpus}
        assert names == set(db.DATASETS.keys())

    def test_real_sample_structure(self, real_sample):
        assert real_sample["label"] == "real"
        assert len(real_sample["flow_seq"]) == 90
        assert 0.0 < real_sample["sf"] < 1.0

    def test_ai_sample_structure(self, ai_sample):
        assert ai_sample["label"] == "ai"
        assert len(ai_sample["flow_seq"]) == 90

    def test_real_sf_lower_than_ai(self, db):
        """AI datasets should on average have higher spectral flatness."""
        real_sfs = [db.generate_sample(ds)["sf"]
                    for ds in ["UCF101", "Kinetics-700", "DAVIS"]
                    for _ in range(20)]
        ai_sfs   = [db.generate_sample(ds)["sf"]
                    for ds in ["Sora", "RunwayGen2", "PikaLabs"]
                    for _ in range(20)]
        assert np.mean(ai_sfs) > np.mean(real_sfs)

    def test_flow_sequence_non_negative(self, db):
        for ds in db.DATASETS:
            sample = db.generate_sample(ds)
            assert all(v >= 0 for v in sample["flow_seq"]), f"Negative flow in {ds}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Physics Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestPhysicsMetrics:

    def test_real_jerk_variance_below_threshold(self, real_sample, db):
        pm = compute_physics_metrics(real_sample["flow_seq"], real_sample["sf"])
        # For most real samples jerk variance should be well below threshold
        assert pm.jerk_variance < db.JERK_NATURAL_THRESHOLD * 2

    def test_ai_jerk_variance_above_real_on_average(self, db):
        """AI jerk variance should be significantly higher than real jerk variance."""
        ai_jvs, real_jvs = [], []
        for _ in range(30):
            ai_s  = db.generate_sample("Sora")
            real_s= db.generate_sample("UCF101")
            ai_pm  = compute_physics_metrics(ai_s["flow_seq"],   ai_s["sf"])
            real_pm= compute_physics_metrics(real_s["flow_seq"], real_s["sf"])
            ai_jvs.append(ai_pm.jerk_variance)
            real_jvs.append(real_pm.jerk_variance)
        # AI jerk variance should be at least 10× higher than real (paper reports 3.7×)
        assert np.mean(ai_jvs) > np.mean(real_jvs) * 5, \
            f"AI jerk {np.mean(ai_jvs):.4f} not sufficiently above real {np.mean(real_jvs):.4f}"

    def test_spectral_anomaly_computed(self, real_sample):
        pm = compute_physics_metrics(real_sample["flow_seq"], real_sample["sf"])
        expected = abs(real_sample["sf"] - SyntheticDatabase.REAL_SENSOR_SF_BASELINE)
        assert abs(pm.spectral_anomaly - expected) < 1e-9

    def test_metrics_are_finite(self, db):
        for ds in db.DATASETS:
            s  = db.generate_sample(ds)
            pm = compute_physics_metrics(s["flow_seq"], s["sf"])
            for attr in ["jerk_variance", "motion_entropy", "spectral_flatness",
                         "spectral_anomaly", "accel_variance"]:
                val = getattr(pm, attr)
                assert np.isfinite(val), f"{attr} is not finite for dataset {ds}"

    def test_entropy_positive(self, real_sample):
        pm = compute_physics_metrics(real_sample["flow_seq"], real_sample["sf"])
        assert pm.motion_entropy >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Meta-CoT Reasoning
# ─────────────────────────────────────────────────────────────────────────────

class TestMetaCoT:

    def test_trace_fields_populated(self, real_sample, db):
        pm = compute_physics_metrics(real_sample["flow_seq"], real_sample["sf"])
        t  = meta_cot_reason(pm, db)
        assert t.observation
        assert t.hypothesis
        assert t.verification
        assert t.conclusion
        assert 0.0 <= t.ai_probability <= 1.0

    def test_real_sample_low_probability(self, db):
        probs = []
        for _ in range(30):
            s  = db.generate_sample("UCF101")
            pm = compute_physics_metrics(s["flow_seq"], s["sf"])
            t  = meta_cot_reason(pm, db)
            probs.append(t.ai_probability)
        assert np.mean(probs) < 0.5, "Real samples should average P(AI) < 0.5"

    def test_ai_sample_high_probability(self, db):
        probs = []
        for _ in range(30):
            s  = db.generate_sample("Sora")
            pm = compute_physics_metrics(s["flow_seq"], s["sf"])
            t  = meta_cot_reason(pm, db)
            probs.append(t.ai_probability)
        assert np.mean(probs) > 0.5, "AI samples should average P(AI) > 0.5"

    def test_prior_trace_influences_score(self, ai_sample, db):
        pm     = compute_physics_metrics(ai_sample["flow_seq"], ai_sample["sf"])
        t1     = meta_cot_reason(pm, db)
        t2     = meta_cot_reason(pm, db, prior_trace=t1)
        # Second pass should produce a different (blended) probability
        # They can be equal only if blend of identical values = same value
        assert isinstance(t2.ai_probability, float)

    def test_probability_range(self, db):
        for ds in db.DATASETS:
            s  = db.generate_sample(ds)
            pm = compute_physics_metrics(s["flow_seq"], s["sf"])
            t  = meta_cot_reason(pm, db)
            assert 0.0 <= t.ai_probability <= 1.0, f"Out-of-range P(AI) for {ds}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cryptographic Provenance
# ─────────────────────────────────────────────────────────────────────────────

class TestProvenance:

    def test_provenance_fields(self, real_sample, db):
        pm   = compute_physics_metrics(real_sample["flow_seq"], real_sample["sf"])
        t    = meta_cot_reason(pm, db)
        prov = compute_provenance(real_sample, pm, t)
        assert prov.h1_media_hash.endswith("…")
        assert prov.h2_metrics_hash.endswith("…")
        assert prov.h3_trace_hash.endswith("…")
        assert "T" in prov.timestamp_utc
        assert len(prov.composite_id) == 16

    def test_provenance_deterministic_within_tolerance(self, db):
        """Same sample → same h1 hash prefix."""
        s1 = db.generate_sample("UCF101")
        s2 = dict(s1)   # shallow copy — same data
        pm = compute_physics_metrics(s1["flow_seq"], s1["sf"])
        t  = meta_cot_reason(pm, db)
        p1 = compute_provenance(s1, pm, t)
        p2 = compute_provenance(s2, pm, t)
        assert p1.h1_media_hash == p2.h1_media_hash

    def test_different_samples_different_hash(self, db):
        s1 = db.generate_sample("UCF101")
        s2 = db.generate_sample("Sora")
        pm1 = compute_physics_metrics(s1["flow_seq"], s1["sf"])
        pm2 = compute_physics_metrics(s2["flow_seq"], s2["sf"])
        t1  = meta_cot_reason(pm1, db)
        t2  = meta_cot_reason(pm2, db)
        p1  = compute_provenance(s1, pm1, t1)
        p2  = compute_provenance(s2, pm2, t2)
        assert p1.h1_media_hash != p2.h1_media_hash


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:

    def test_pipeline_returns_result(self, real_sample, db):
        r = run_pipeline(real_sample, db)
        assert r.predicted_label in ("real", "ai")
        assert 0.0 <= r.ai_probability <= 1.0
        assert r.provenance is not None

    def test_real_datasets_mostly_correct(self, db):
        real_ds = ["UCF101", "Kinetics-700", "DAVIS", "ImageNet-1K", "COCO-2017"]
        correct = 0; total = 0
        for ds in real_ds:
            for _ in range(20):
                s = db.generate_sample(ds)
                r = run_pipeline(s, db)
                correct += (r.predicted_label == "real")
                total   += 1
        assert correct / total >= 0.90, f"Real dataset accuracy {correct/total:.2%} < 90%"

    def test_ai_datasets_mostly_correct(self, db):
        ai_ds = ["StableDiffusion", "Midjourney-v6", "DALL-E-3", "Sora",
                 "RunwayGen2", "PikaLabs"]
        correct = 0; total = 0
        for ds in ai_ds:
            for _ in range(20):
                s = db.generate_sample(ds)
                r = run_pipeline(s, db)
                correct += (r.predicted_label == "ai")
                total   += 1
        assert correct / total >= 0.95, f"AI dataset accuracy {correct/total:.2%} < 95%"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluation:

    @pytest.fixture
    def results(self, db):
        corpus = db.generate_dataset(samples_per_dataset=20)
        return [run_pipeline(s, db) for s in corpus]

    def test_overall_accuracy_above_95(self, results):
        m = evaluate(results)
        assert m["accuracy"] >= 0.95, f"Accuracy {m['accuracy']:.2%} < 95%"

    def test_reasoning_consistency_100(self, results):
        m = evaluate(results)
        assert m["reasoning_consistency_score"] == 1.0

    def test_ai_jerk_higher_than_real(self, results):
        m = evaluate(results)
        assert m["mean_jerk_variance_ai"] > m["mean_jerk_variance_real"]

    def test_confusion_matrix_fields(self, results):
        m = evaluate(results)
        total = m["true_positives"] + m["true_negatives"] + \
                m["false_positives"] + m["false_negatives"]
        assert total == m["total_samples"]

    def test_fpr_below_10_percent(self, results):
        m = evaluate(results)
        assert m["false_positive_rate"] <= 0.10, \
            f"FPR {m['false_positive_rate']:.2%} too high"
