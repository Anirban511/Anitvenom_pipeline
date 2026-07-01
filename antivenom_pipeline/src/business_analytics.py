#!/usr/bin/env python3
"""
================================================================================
BUSINESS ANALYTICS LAYER
================================================================================
Turns the raw pipeline output (sequences + scores) into the decision metrics
a medicine-making company actually cares about when choosing a design method:

    - Cost per viable candidate          (Rs / $ per usable design)
    - Latency per candidate              (seconds of wall-clock + compute)
    - Throughput                         (candidates designable per GPU-hour)
    - Quality parity                     (how close LLM quality is to the
                                          structure-based gold standard)
    - Screening-funnel economics         (cost to reach N viable candidates)

HONEST SCOPE
------------
The QUALITY numbers flow from the upstream scorer (which, in the current
prototype, uses synthetic placeholder values for ProteinMPNN/AlphaFold/docking).
The COST and LATENCY numbers, however, are driven by REAL, defensible facts
about the two methods:

    * ProtGPT2 (LLM):     needs ONLY a sequence -> no structure step.
    * ProteinMPNN:        needs a 3D backbone for EVERY candidate -> an
                          expensive structure-prediction/preparation step
                          dominates its latency and cost.

That structural dependency is the genuine business lever. The thesis is NOT
"the LLM makes better antivenoms" — it is "the LLM reaches COMPARABLE quality
at a fraction of the cost and latency, making it the better tool for
high-throughput early-stage screening."

All cost/time constants below are EXPLICIT and editable. They are modelling
assumptions, not measurements — change them to match your own benchmarks and
re-run. Every number the report prints can be traced to a constant here.
================================================================================
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# ASSUMPTION BLOCK  (edit these to match your real benchmarks, then re-run)
# ==============================================================================
@dataclass
class CostModel:
    """
    Editable economic assumptions. Defaults are order-of-magnitude estimates
    chosen to reflect the *structural dependency* difference between methods.
    Replace with measured values from your own runs for a real business case.
    """
    # --- Compute pricing ---
    gpu_cost_per_hour_usd: float = 2.50      # typical cloud GPU (e.g. T4/A10 tier)

    # --- Per-candidate LATENCY (seconds of GPU wall-clock) ---
    # LLM path: generation only (no structure needed)
    llm_seconds_per_candidate: float = 4.0
    # Structure-based path: must predict/prepare a 3D backbone, THEN design
    mpnn_structure_seconds_per_candidate: float = 90.0   # the expensive step
    mpnn_design_seconds_per_candidate: float = 6.0       # design once structure exists

    # --- Quality gate ---
    # Fraction of generated candidates that pass the viability score threshold.
    # (Structure-based designs pass at a higher rate; LLM casts a wider, noisier net.)
    llm_viability_pass_rate: float = 0.40
    mpnn_viability_pass_rate: float = 0.65

    # --- Currency ---
    usd_to_inr: float = 83.0

    def llm_total_seconds(self) -> float:
        return self.llm_seconds_per_candidate

    def mpnn_total_seconds(self) -> float:
        return self.mpnn_structure_seconds_per_candidate + self.mpnn_design_seconds_per_candidate


# ==============================================================================
# CORE ANALYTICS
# ==============================================================================
@dataclass
class MethodEconomics:
    """Computed economics for ONE design method."""
    method: str
    candidates_generated: int
    seconds_per_candidate: float
    viability_pass_rate: float

    # derived
    viable_candidates: float = 0.0
    total_compute_seconds: float = 0.0
    cost_per_candidate_usd: float = 0.0
    cost_per_viable_usd: float = 0.0
    throughput_per_gpu_hour: float = 0.0
    mean_quality_score: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class BusinessAnalytics:
    """
    Converts pipeline output into pharma-facing decision metrics.
    """

    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cm = cost_model or CostModel()

    # ----- economics for a single method -----
    def compute_method_economics(
        self,
        method: str,
        candidates_generated: int,
        seconds_per_candidate: float,
        viability_pass_rate: float,
        mean_quality_score: Optional[float] = None,
    ) -> MethodEconomics:
        cm = self.cm
        cost_per_sec = cm.gpu_cost_per_hour_usd / 3600.0

        total_seconds = candidates_generated * seconds_per_candidate
        cost_per_candidate = seconds_per_candidate * cost_per_sec
        viable = max(candidates_generated * viability_pass_rate, 1e-9)
        cost_per_viable = (total_seconds * cost_per_sec) / viable
        throughput = 3600.0 / seconds_per_candidate if seconds_per_candidate > 0 else 0.0

        return MethodEconomics(
            method=method,
            candidates_generated=candidates_generated,
            seconds_per_candidate=seconds_per_candidate,
            viability_pass_rate=viability_pass_rate,
            viable_candidates=round(viable, 2),
            total_compute_seconds=round(total_seconds, 2),
            cost_per_candidate_usd=round(cost_per_candidate, 5),
            cost_per_viable_usd=round(cost_per_viable, 5),
            throughput_per_gpu_hour=round(throughput, 1),
            mean_quality_score=mean_quality_score,
        )

    # ----- full comparative report -----
    def build_report(
        self,
        n_llm: int,
        n_mpnn: int,
        llm_quality: Optional[float] = None,
        mpnn_quality: Optional[float] = None,
    ) -> Dict:
        cm = self.cm

        llm_econ = self.compute_method_economics(
            "LLM (ProtGPT2)", n_llm, cm.llm_total_seconds(),
            cm.llm_viability_pass_rate, llm_quality,
        )
        mpnn_econ = self.compute_method_economics(
            "ProteinMPNN", n_mpnn, cm.mpnn_total_seconds(),
            cm.mpnn_viability_pass_rate, mpnn_quality,
        )

        # --- headline comparative levers ---
        latency_speedup = (
            mpnn_econ.seconds_per_candidate / llm_econ.seconds_per_candidate
            if llm_econ.seconds_per_candidate else 0.0
        )
        cost_per_viable_ratio = (
            mpnn_econ.cost_per_viable_usd / llm_econ.cost_per_viable_usd
            if llm_econ.cost_per_viable_usd else 0.0
        )
        throughput_gain = (
            llm_econ.throughput_per_gpu_hour / mpnn_econ.throughput_per_gpu_hour
            if mpnn_econ.throughput_per_gpu_hour else 0.0
        )

        quality_parity = None
        if llm_quality is not None and mpnn_quality is not None and mpnn_quality:
            quality_parity = round(100.0 * llm_quality / mpnn_quality, 1)

        # --- screening-funnel scenario: cost to reach 100 viable candidates ---
        target_viable = 100
        llm_needed = target_viable / max(cm.llm_viability_pass_rate, 1e-9)
        mpnn_needed = target_viable / max(cm.mpnn_viability_pass_rate, 1e-9)
        cost_per_sec = cm.gpu_cost_per_hour_usd / 3600.0
        llm_funnel_cost = llm_needed * cm.llm_total_seconds() * cost_per_sec
        mpnn_funnel_cost = mpnn_needed * cm.mpnn_total_seconds() * cost_per_sec
        llm_funnel_hours = (llm_needed * cm.llm_total_seconds()) / 3600.0
        mpnn_funnel_hours = (mpnn_needed * cm.mpnn_total_seconds()) / 3600.0

        report = {
            "assumptions": asdict(cm),
            "per_method": {
                "llm": llm_econ.to_dict(),
                "mpnn": mpnn_econ.to_dict(),
            },
            "headline_levers": {
                "latency_speedup_x": round(latency_speedup, 1),
                "cost_per_viable_ratio_x": round(cost_per_viable_ratio, 1),
                "throughput_gain_x": round(throughput_gain, 1),
                "quality_parity_pct": quality_parity,
            },
            "funnel_scenario_100_viable": {
                "target_viable_candidates": target_viable,
                "llm": {
                    "candidates_to_generate": round(llm_needed),
                    "gpu_hours": round(llm_funnel_hours, 2),
                    "cost_usd": round(llm_funnel_cost, 2),
                    "cost_inr": round(llm_funnel_cost * cm.usd_to_inr, 2),
                },
                "mpnn": {
                    "candidates_to_generate": round(mpnn_needed),
                    "gpu_hours": round(mpnn_funnel_hours, 2),
                    "cost_usd": round(mpnn_funnel_cost, 2),
                    "cost_inr": round(mpnn_funnel_cost * cm.usd_to_inr, 2),
                },
                "savings_usd": round(mpnn_funnel_cost - llm_funnel_cost, 2),
                "savings_pct": round(
                    100.0 * (mpnn_funnel_cost - llm_funnel_cost) / mpnn_funnel_cost, 1
                ) if mpnn_funnel_cost else 0.0,
            },
        }
        return report

    # ----- human-readable console summary -----
    def print_report(self, report: Dict) -> None:
        h = report["headline_levers"]
        f = report["funnel_scenario_100_viable"]
        llm = report["per_method"]["llm"]
        mpnn = report["per_method"]["mpnn"]

        print("\n" + "=" * 74)
        print("  BUSINESS ANALYTICS  —  Antivenom Design Method Selection")
        print("=" * 74)

        print("\n  PER-METHOD ECONOMICS")
        print("  " + "-" * 70)
        print(f"  {'Metric':<34}{'LLM (ProtGPT2)':>18}{'ProteinMPNN':>18}")
        print("  " + "-" * 70)
        print(f"  {'Seconds / candidate':<34}{llm['seconds_per_candidate']:>18}{mpnn['seconds_per_candidate']:>18}")
        print(f"  {'Viability pass rate':<34}{llm['viability_pass_rate']:>18}{mpnn['viability_pass_rate']:>18}")
        print(f"  {'Cost / candidate (USD)':<34}{llm['cost_per_candidate_usd']:>18}{mpnn['cost_per_candidate_usd']:>18}")
        print(f"  {'Cost / VIABLE candidate (USD)':<34}{llm['cost_per_viable_usd']:>18}{mpnn['cost_per_viable_usd']:>18}")
        print(f"  {'Throughput / GPU-hour':<34}{llm['throughput_per_gpu_hour']:>18}{mpnn['throughput_per_gpu_hour']:>18}")
        if llm.get("mean_quality_score") is not None:
            print(f"  {'Mean quality score':<34}{llm['mean_quality_score']:>18}{mpnn['mean_quality_score']:>18}")

        print("\n  HEADLINE LEVERS (why a pharma team would care)")
        print("  " + "-" * 70)
        print(f"  LLM latency speed-up vs ProteinMPNN .......... {h['latency_speedup_x']}x faster")
        print(f"  LLM cost-per-viable advantage ............... {h['cost_per_viable_ratio_x']}x cheaper")
        print(f"  LLM throughput gain ......................... {h['throughput_gain_x']}x more designs / GPU-hour")
        if h["quality_parity_pct"] is not None:
            print(f"  Quality parity (LLM as % of gold standard) .. {h['quality_parity_pct']}%")

        print("\n  FUNNEL SCENARIO — reach 100 viable candidates")
        print("  " + "-" * 70)
        print(f"  LLM ....... generate {f['llm']['candidates_to_generate']:>5}  |  "
              f"{f['llm']['gpu_hours']:>6} GPU-hrs  |  ${f['llm']['cost_usd']:>8}")
        print(f"  MPNN ...... generate {f['mpnn']['candidates_to_generate']:>5}  |  "
              f"{f['mpnn']['gpu_hours']:>6} GPU-hrs  |  ${f['mpnn']['cost_usd']:>8}")
        print(f"  >>> LLM saves ${f['savings_usd']} ({f['savings_pct']}%) to reach the same 100 viable hits")
        print("=" * 74 + "\n")


# ==============================================================================
# helper: derive analytics inputs from a pipeline results.json
# ==============================================================================
def analytics_from_results(results: Dict, cost_model: Optional[CostModel] = None) -> Dict:
    """
    Accepts the pipeline's results dict and produces the business report.
    Pulls candidate counts and (if present) mean quality scores per method.
    """
    llm_seqs = results.get("llm_sequences", [])
    mpnn_seqs = results.get("mpnn_sequences", [])
    n_llm = max(len(llm_seqs), 1)
    n_mpnn = max(len(mpnn_seqs), 1)

    def _mean_quality(seqs):
        vals = [s.get("composite_score") or s.get("avg_plddt") for s in seqs
                if (s.get("composite_score") or s.get("avg_plddt")) is not None]
        return round(float(np.mean(vals)), 1) if vals else None

    llm_q = _mean_quality(llm_seqs)
    mpnn_q = _mean_quality(mpnn_seqs)

    ba = BusinessAnalytics(cost_model)
    report = ba.build_report(n_llm, n_mpnn, llm_q, mpnn_q)
    ba.print_report(report)
    return report


# ==============================================================================
# standalone demo
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ba = BusinessAnalytics()
    # Demo with 5 LLM + 5 MPNN candidates and illustrative quality scores
    demo = ba.build_report(n_llm=5, n_mpnn=5, llm_quality=69.9, mpnn_quality=76.2)
    ba.print_report(demo)
    print("Demo report keys:", list(demo.keys()))
