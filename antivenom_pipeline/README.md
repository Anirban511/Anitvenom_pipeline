# Antivenom Protein Design Pipeline

An end-to-end pipeline that explores whether a **language model (ProtGPT2)** can
match a **structure-based method (ProteinMPNN)** for designing proteins that bind
snake-venom toxins — wrapped in production-engineering infrastructure (retry,
caching, validation, config, logging) and covered by a unit-test suite.

> **Project type:** guided research prototype (supervised by Prof. Riddhiman Dhar,
> IIT Kharagpur). The orchestration, biophysical scoring, business-analytics, and
> infrastructure layers are real and tested. The deep-learning components
> (ProteinMPNN, AlphaFold, docking) currently use **synthetic placeholder values**
> behind real interfaces — see [Real vs. Prototype](#real-vs-prototype) below.

---

## Quickstart

```bash
pip install -r requirements.txt     # or: make install
python run_all.py                   # or: make run
pytest tests/ -v                    # or: make test  (25 tests)
```

No API key required. Without `torch`/`transformers` installed, sequence
generation automatically falls back to a dependency-free generator, so the
pipeline still runs end-to-end.

---

## Architecture

```
run_all.py  (orchestrator, CLI)
  │
  ├─ src/main_pipeline.py        Stage controller + domain logic
  │     ├─ PDBHandler            Stage 1: fetch toxin structure (retry + cache)
  │     ├─ ProtGPT2 generator    Stage 2: LLM sequence generation
  │     ├─ ProteinMPNN           Stage 3: structure-based generation
  │     ├─ AlphaFoldPredictor    Stage 4: fold + pLDDT confidence
  │     └─ ProteinAnalyzer       Stage 5: features, scoring, comparison
  │
  ├─ src/docking_analysis.py     Binding-energy estimation
  ├─ src/metrics_analysis.py     Composite scoring + method comparison
  ├─ src/business_analytics.py   Stage 7: cost / latency / throughput economics
  │
  └─ src/infrastructure.py       Cross-cutting: retry, cache, config, validation, logging
```

The pipeline is a **layered system**: a domain core, a business-analytics layer
on top, and a cross-cutting infrastructure layer that the core calls into.

---

## Engineering features

These are real, tested, and integrated into the pipeline. Each exists to solve a
specific failure mode — see `docs`/code docstrings for the full rationale.

| Feature | Problem it solves | Where |
|---|---|---|
| **Retry + exponential backoff** | Transient network failures on PDB download (a real HTTP 403 hit during dev) | `infrastructure.retry_with_backoff`, used by `PDBHandler._fetch` |
| **Content-addressed disk cache** | Redundant re-downloads / re-generation across runs; stale reads made structurally impossible by hashing inputs | `infrastructure.DiskCache`, used by `PDBHandler` |
| **Typed + validated config** | Scattered magic numbers; silent misconfiguration. Fails fast if scoring weights don't sum to 1.0 | `infrastructure.PipelineConfig`, env-overridable via `PIPELINE_*` |
| **Fail-fast input validation** | Malformed model output polluting analytics; bad PDB IDs caught at the boundary | `infrastructure.validate_pdb_id`, `clean_sequence` |
| **Structured logging** | Unreproducible multi-stage failures | `infrastructure.get_logger` |
| **Unit test suite (25 tests)** | Proof the guarantees hold; safe refactoring | `tests/test_pipeline.py` |

Run the infrastructure smoke test directly:

```bash
python src/infrastructure.py
```

---

## Configuration

All knobs live in `src/infrastructure.py::PipelineConfig` and can be overridden by
environment variables (`PIPELINE_<FIELD>`), e.g.:

```bash
PIPELINE_NUM_SEQUENCES=10 PIPELINE_TEMPERATURE=0.9 python run_all.py
PIPELINE_ENABLE_CACHE=false python run_all.py     # or: python run_all.py --no-cache
```

Config is validated at startup — invalid values raise immediately instead of
failing midway through a run.

---

## Outputs

Written to `results/`:

- `results.json` — per-sequence metrics, method comparison, cache stats
- `business_analytics.json` — cost-per-viable-candidate, latency speedup,
  throughput, quality parity, and a "reach 100 viable candidates" funnel scenario

---

## Real vs. Prototype

Honesty matters more than a polished claim. Here is exactly what is real:

| Stage | Status | Notes |
|---|---|---|
| PDB download + validation | **REAL** | Live RCSB fetch, retry, cache, ATOM validation |
| ProtGPT2 generation | **REAL\*** | Real transformer inference if `torch`/`transformers` present; dependency-free fallback otherwise |
| Cysteine / hydrophobicity / charge features | **REAL** | Genuine biophysical calculations |
| Composite scoring + comparison | **REAL** | Weighted ranking, operates on whatever inputs it's given |
| Business analytics (cost/latency) | **REAL (model)** | Driven by explicit, editable assumptions in `CostModel`, grounded in the real fact that the LLM needs no structure step |
| Infrastructure layer | **REAL** | 25 passing unit tests |
| ProteinMPNN generation | **PLACEHOLDER** | Returns random sequences; real weights not loaded |
| AlphaFold pLDDT | **PLACEHOLDER** | `np.random.normal`; no folding performed |
| Docking binding energy | **MIXED** | Real biophysics-style decomposition; top-level value still synthetic |

**To productionize:** swap the three placeholders for real
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN), ESMFold/AlphaFold, and
AutoDock Vina behind the existing interfaces — the surrounding orchestration,
scoring, analytics, and infrastructure need no changes.

---

## Project layout

```
antivenom_pipeline/
├─ run_all.py            # single entry point
├─ config.yaml           # human-editable defaults
├─ requirements.txt
├─ Makefile
├─ src/
│  ├─ main_pipeline.py
│  ├─ docking_analysis.py
│  ├─ metrics_analysis.py
│  ├─ business_analytics.py
│  ├─ llm_sequence_generator_protgpt2.py
│  └─ infrastructure.py
└─ tests/
   ├─ conftest.py
   └─ test_pipeline.py
```

---

## License / attribution

Research prototype, IIT Kharagpur. ProtGPT2 (Ferruz et al., 2022),
ProteinMPNN (Dauparas et al., 2022), AlphaFold (Jumper et al., 2021).
