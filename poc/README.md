# MotionSemanticProvenanceAI — POC

**Meta Chain-of-Thought Guided Physics-Semantic Provenance Verification**  
*Proof-of-Concept implementing the pipeline from Indranill Datta, 2026 (Pre-Print)*

---

## Overview

This POC implements the full **9-stage MotionSemanticProvenanceAI pipeline** (Algorithm 1)
and **11-dataset synthetic benchmark** (Table III) from the paper, runnable on any CPU
without GPU or real video data.

---

## Files

| File | Purpose |
|------|---------|
| `motion_semantic_provenance.py` | Full pipeline implementation |
| `test_semantic_provenance.py`   | 27 pytest unit + integration tests |

---

## Architecture

```
SyntheticDatabase           →  11 datasets (5 real, 6 AI-generated)
    ↓
compute_physics_metrics()   →  Stage ②–⑥: jerk, entropy, spectral flatness
    ↓
meta_cot_reason()           →  Stage ⑦:  Meta-CoT 4-step reasoning trace
    ↓
compute_provenance()        →  Stage ⑧:  SHA-256 (media + metrics + trace)
    ↓
run_pipeline()              →  Stage ⑨:  P(AI) ∈ [0,1]  +  ProvenanceRecord
    ↓
evaluate()                  →  Table IV metrics
```

---

## Synthetic Databases (Table III mapping)

| Dataset           | Type  | Label | Jerk scale | SF mean |
|-------------------|-------|-------|-----------|---------|
| UCF101            | Video | Real  | 0.8       | 0.38    |
| Kinetics-700      | Video | Real  | 0.9       | 0.41    |
| DAVIS             | Video | Real  | 0.7       | 0.36    |
| ImageNet-1K       | Image | Real  | 0.5       | 0.43    |
| COCO-2017         | Image | Real  | 0.6       | 0.44    |
| Stable Diffusion  | Image | AI    | 4.2       | 0.79    |
| Midjourney v6     | Image | AI    | 4.8       | 0.82    |
| DALL-E 3          | Image | AI    | 4.0       | 0.77    |
| Sora              | Video | AI    | 5.5       | 0.85    |
| Runway Gen-2      | Video | AI    | 5.0       | 0.83    |
| Pika Labs 1.0     | Video | AI    | 4.9       | 0.81    |

---

## Run the POC

```bash
pip install numpy scipy scikit-learn
python motion_semantic_provenance.py
```

Expected output: 550 samples verified, overall accuracy ~99%, FPR ~1.6%.

---

## Run Tests

```bash
pip install pytest
python -m pytest test_semantic_provenance.py -v
```

Expected: **27 passed**.

---

## POC Results (Table IV)

| Metric | Result |
|--------|--------|
| Accuracy | 99.27% |
| False Positive Rate | 1.60% |
| Reasoning Consistency Score | 100% |
| Jerk Variance Divergence | 0.498 |
| Spectral Divergence | 0.337 |
| Mean Jerk Var (Real) | 0.013 |
| Mean Jerk Var (AI) | 0.511 |

---

## Test Coverage (27 tests)

| Class | Tests | Focus |
|-------|-------|-------|
| `TestSyntheticDatabase` | 6 | DB generation, labels, flow values |
| `TestPhysicsMetrics` | 5 | Jerk/entropy/SF computation |
| `TestMetaCoT` | 5 | Trace fields, probability range |
| `TestProvenance` | 3 | Hash determinism, uniqueness |
| `TestPipeline` | 3 | End-to-end integration |
| `TestEvaluation` | 5 | Accuracy, FPR, RCS, TP/TN |
