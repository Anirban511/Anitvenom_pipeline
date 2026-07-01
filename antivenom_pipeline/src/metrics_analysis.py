#!/usr/bin/env python3
"""
COMPARATIVE METRICS AND ANALYSIS
Detailed comparison of LLM vs ProteinMPNN sequences

Key Metrics:
1. Structural Quality (pLDDT, secondary structure)
2. Stability (cysteine bonds, hydrophobicity, charge)
3. Biological Feasibility
4. Docking Performance
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """Complete metrics for a sequence"""
    sequence_id: str
    method: str  # LLM or ProteinMPNN
    sequence: str
    
    # Structural quality
    avg_plddt: float
    max_plddt: float
    plddt_stability: float  # std dev of pLDDT scores
    secondary_structure_score: float  # Predicted % structured regions
    
    # Stability
    disulfide_bonds: int
    cysteine_pairing_efficiency: float  # 0-1, how well cysteines pair
    net_charge: int
    charge_distribution_score: float
    hydrophobicity: float
    hydrophobic_patches: int
    
    # Composition
    length: int
    aromatic_content: float
    polar_content: float
    
    # Docking
    binding_energy: float = None
    binding_confidence: float = None
    hydrogen_bonds_per_aa: float = None
    
    # Overall scores
    structure_score: float = None  # 0-100
    stability_score: float = None  # 0-100
    dockability_score: float = None  # 0-100
    composite_score: float = None  # Weighted average


class SequenceAnalyzer:
    """Detailed sequence analysis"""
    
    AMINO_ACIDS_AROMATIC = set('FWY')
    AMINO_ACIDS_POLAR = set('STNQC')
    AMINO_ACIDS_CHARGED_POS = set('KRH')
    AMINO_ACIDS_CHARGED_NEG = set('DE')
    AMINO_ACIDS_HYDROPHOBIC = set('AILMFVP')
    
    HYDROPATHY_INDEX = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    
    @classmethod
    def analyze_complete(
        cls,
        sequence: str,
        method: str,
        seq_id: str,
        plddt_scores: List[float],
        binding_energy: float = None
    ) -> ComparisonMetrics:
        """Comprehensive sequence analysis"""
        
        # Structural quality
        avg_plddt = np.mean(plddt_scores)
        max_plddt = np.max(plddt_scores)
        plddt_std = np.std(plddt_scores)
        ss_score = cls._predict_secondary_structure_score(plddt_scores)
        
        # Stability analysis
        cys_count, cys_pairs, cys_eff = cls._analyze_disulfides(sequence)
        net_charge = cls._calculate_net_charge(sequence)
        charge_dist = cls._analyze_charge_distribution(sequence)
        hydro = cls._calculate_hydrophobicity(sequence)
        hydro_patches = cls._count_hydrophobic_patches(sequence)
        
        # Composition
        aromatic = sum(1 for aa in sequence if aa in cls.AMINO_ACIDS_AROMATIC) / len(sequence)
        polar = sum(1 for aa in sequence if aa in cls.AMINO_ACIDS_POLAR) / len(sequence)
        
        # Score calculations
        structure_score = cls._calculate_structure_score(avg_plddt, plddt_std, ss_score)
        stability_score = cls._calculate_stability_score(
            cys_eff, net_charge, charge_dist, hydro, len(sequence)
        )
        
        # Docking metrics
        hbonds_per_aa = None
        dockability_score = None
        binding_confidence = None
        
        if binding_energy is not None:
            binding_confidence = cls._binding_to_confidence(binding_energy)
            dockability_score = cls._calculate_dockability_score(
                structure_score, stability_score, aromatic, binding_energy
            )
            hbonds_per_aa = cls._estimate_hbonds_per_aa(sequence)
        
        # Composite score
        composite_score = cls._calculate_composite_score(
            structure_score, stability_score, dockability_score
        )
        
        return ComparisonMetrics(
            sequence_id=seq_id,
            method=method,
            sequence=sequence,
            avg_plddt=float(avg_plddt),
            max_plddt=float(max_plddt),
            plddt_stability=float(plddt_std),
            secondary_structure_score=float(ss_score),
            disulfide_bonds=cys_pairs,
            cysteine_pairing_efficiency=float(cys_eff),
            net_charge=net_charge,
            charge_distribution_score=float(charge_dist),
            hydrophobicity=float(hydro),
            hydrophobic_patches=hydro_patches,
            length=len(sequence),
            aromatic_content=float(aromatic),
            polar_content=float(polar),
            binding_energy=binding_energy,
            binding_confidence=binding_confidence,
            hydrogen_bonds_per_aa=hbonds_per_aa,
            structure_score=float(structure_score),
            stability_score=float(stability_score),
            dockability_score=float(dockability_score) if dockability_score else None,
            composite_score=float(composite_score) if composite_score else None
        )
    
    @staticmethod
    def _predict_secondary_structure_score(plddt_scores: List[float]) -> float:
        """Higher confidence = more likely to form secondary structure"""
        high_conf = sum(1 for p in plddt_scores if p > 80)
        return high_conf / len(plddt_scores) if plddt_scores else 0
    
    @staticmethod
    def _analyze_disulfides(sequence: str) -> Tuple[int, int, float]:
        """Analyze cysteine bonds and pairing efficiency"""
        cys_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
        cys_count = len(cys_positions)
        
        if cys_count < 2:
            return cys_count, 0, 0.0
        
        # Optimal cysteine distance for disulfide bond: 20-60 residues (typical)
        pairs = []
        remaining = cys_positions.copy()
        
        while len(remaining) >= 2:
            c1 = remaining.pop(0)
            # Find best partner (distance-wise)
            best_idx = 0
            best_distance = abs(remaining[0] - c1)
            
            for i, c2 in enumerate(remaining[1:], 1):
                dist = abs(c2 - c1)
                if 20 <= dist <= 60:  # Optimal range
                    if dist < best_distance:
                        best_idx = i
                        best_distance = dist
            
            pairs.append((c1, remaining.pop(best_idx)))
        
        efficiency = len(pairs) / (cys_count // 2) if cys_count >= 2 else 0
        
        return cys_count, len(pairs), efficiency
    
    @staticmethod
    def _calculate_net_charge(sequence: str) -> int:
        """Net charge at physiological pH"""
        pos = sum(1 for aa in sequence if aa in 'KRH')
        neg = sum(1 for aa in sequence if aa in 'DE')
        return pos - neg
    
    @staticmethod
    def _analyze_charge_distribution(sequence: str) -> float:
        """Score of how evenly charges are distributed"""
        charged_positions = [
            (i, 1 if aa in 'KRH' else -1)
            for i, aa in enumerate(sequence)
            if aa in 'KRHDE'
        ]
        
        if len(charged_positions) < 2:
            return 1.0  # Perfect for no/one charge
        
        # Calculate local charge clustering
        max_window_charge = 0
        window_size = 10
        for i in range(len(sequence) - window_size):
            window_charge = sum(
                1 if pos in range(i, i+window_size) and charge > 0 else 
                -1 if pos in range(i, i+window_size) and charge < 0 else 0
                for pos, charge in charged_positions
            )
            max_window_charge = max(max_window_charge, abs(window_charge))
        
        # Lower clustering is better
        return max(0, 1.0 - max_window_charge / len(sequence))
    
    @staticmethod
    def _calculate_hydrophobicity(sequence: str) -> float:
        """Kyte-Doolittle hydrophobicity index"""
        hydro_values = [
            SequenceAnalyzer.HYDROPATHY_INDEX.get(aa, 0)
            for aa in sequence
        ]
        return np.mean(hydro_values) if hydro_values else 0
    
    @staticmethod
    def _count_hydrophobic_patches(sequence: str, window: int = 7) -> int:
        """Count significant hydrophobic regions"""
        patches = 0
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i+window]
            hydro_content = sum(
                1 for aa in window_seq
                if SequenceAnalyzer.HYDROPATHY_INDEX.get(aa, 0) > 1.5
            )
            if hydro_content >= 4:  # At least 50% hydrophobic
                patches += 1
        return patches
    
    @staticmethod
    def _estimate_hbonds_per_aa(sequence: str) -> float:
        """Estimate hydrogen bonding capacity"""
        donors = sum(1 for aa in sequence if aa in 'NQSTY')
        acceptors = sum(1 for aa in sequence if aa in 'DEQNSTY')
        potential = (donors + acceptors) / len(sequence) if sequence else 0
        return potential
    
    @staticmethod
    def _calculate_structure_score(avg_plddt: float, plddt_std: float, ss_score: float) -> float:
        """
        Structure quality score (0-100)
        - High pLDDT is good
        - Low variation is good
        - High secondary structure is good
        """
        plddt_component = min(avg_plddt / 100 * 50, 50)  # 0-50
        stability_component = max(0, (25 - plddt_std) / 25 * 20)  # 0-20
        ss_component = ss_score * 30  # 0-30
        
        return plddt_component + stability_component + ss_component
    
    @staticmethod
    def _calculate_stability_score(
        cys_eff: float,
        net_charge: int,
        charge_dist: float,
        hydro: float,
        seq_len: int
    ) -> float:
        """
        Stability score (0-100)
        - Good disulfide bonding
        - Balanced charge
        - Good hydrophobic/hydrophilic balance
        """
        cys_component = cys_eff * 30  # 0-30
        charge_component = (1 - abs(net_charge) / seq_len) * 35  # 0-35
        hydro_component = (1 - abs(hydro) / 4.5) * 35  # 0-35 (optimal around 0)
        
        return cys_component + charge_component + hydro_component
    
    @staticmethod
    def _calculate_dockability_score(
        structure_score: float,
        stability_score: float,
        aromatic_content: float,
        binding_energy: float = None
    ) -> float:
        """
        Dockability score (0-100)
        Ability to bind toxins
        """
        struct_component = structure_score * 0.3
        stability_component = stability_score * 0.3
        aromatic_component = min(aromatic_content, 0.15) / 0.15 * 20  # 0-20
        
        if binding_energy is not None:
            # More negative binding energy = better binding
            energy_component = max(0, min(-binding_energy / 10 * 20, 20))
        else:
            energy_component = 10
        
        return struct_component + stability_component + aromatic_component + energy_component
    
    @staticmethod
    def _calculate_composite_score(
        structure_score: float,
        stability_score: float,
        dockability_score: float = None
    ) -> float:
        """Weighted composite score"""
        if dockability_score is None:
            # Without docking data
            return (structure_score * 0.5 + stability_score * 0.5)
        
        # With docking data
        return (
            structure_score * 0.35 +
            stability_score * 0.35 +
            dockability_score * 0.3
        )
    
    @staticmethod
    def _binding_to_confidence(binding_energy: float) -> float:
        """Convert binding energy to confidence (0-1)"""
        # More negative = better binding
        # -8 kcal/mol is very good, +2 is poor
        confidence = max(0, min(1, (-binding_energy - 2) / 10))
        return confidence


class ComparisonReport:
    """Generate comparison reports"""
    
    @staticmethod
    def compare_methods(llm_metrics: List[ComparisonMetrics], 
                       mpnn_metrics: List[ComparisonMetrics]) -> Dict:
        """Generate comprehensive comparison"""
        
        def stats_dict(metrics_list):
            if not metrics_list:
                return {}
            
            avg_pldd = [m.avg_plddt for m in metrics_list]
            struct = [m.structure_score for m in metrics_list]
            stab = [m.stability_score for m in metrics_list]
            comp = [m.composite_score for m in metrics_list]
            
            return {
                'count': len(metrics_list),
                'avg_plddt': {
                    'mean': float(np.mean(avg_pldd)),
                    'std': float(np.std(avg_pldd)),
                    'range': [float(np.min(avg_pldd)), float(np.max(avg_pldd))]
                },
                'structure_score': {
                    'mean': float(np.mean(struct)),
                    'std': float(np.std(struct))
                },
                'stability_score': {
                    'mean': float(np.mean(stab)),
                    'std': float(np.std(stab))
                },
                'composite_score': {
                    'mean': float(np.mean(comp)),
                    'std': float(np.std(comp))
                }
            }
        
        report = {
            'llm': stats_dict(llm_metrics),
            'proteinmpnn': stats_dict(mpnn_metrics),
            'interpretation': ComparisonReport._interpret_results(llm_metrics, mpnn_metrics)
        }
        
        return report
    
    @staticmethod
    def _interpret_results(llm_metrics: List[ComparisonMetrics],
                          mpnn_metrics: List[ComparisonMetrics]) -> Dict:
        """Interpret which method performs better"""
        
        if not llm_metrics or not mpnn_metrics:
            return {}
        
        llm_struct = np.mean([m.structure_score for m in llm_metrics])
        mpnn_struct = np.mean([m.structure_score for m in mpnn_metrics])
        
        llm_stab = np.mean([m.stability_score for m in llm_metrics])
        mpnn_stab = np.mean([m.stability_score for m in mpnn_metrics])
        
        interpretation = {
            'structure_winner': 'ProteinMPNN' if mpnn_struct > llm_struct else 'LLM',
            'structure_margin': abs(mpnn_struct - llm_struct),
            'stability_winner': 'ProteinMPNN' if mpnn_stab > llm_stab else 'LLM',
            'stability_margin': abs(mpnn_stab - llm_stab),
            'overall_recommendation': ComparisonReport._recommend(llm_metrics, mpnn_metrics)
        }
        
        return interpretation
    
    @staticmethod
    def _recommend(llm_metrics: List[ComparisonMetrics],
                  mpnn_metrics: List[ComparisonMetrics]) -> str:
        """Provide recommendation based on results"""
        
        llm_comp = [m.composite_score for m in llm_metrics if m.composite_score]
        mpnn_comp = [m.composite_score for m in mpnn_metrics if m.composite_score]
        
        if not llm_comp or not mpnn_comp:
            return "Insufficient data for recommendation"
        
        llm_avg = np.mean(llm_comp)
        mpnn_avg = np.mean(mpnn_comp)
        
        diff_percent = abs(mpnn_avg - llm_avg) / max(llm_avg, mpnn_avg) * 100
        
        if diff_percent < 5:
            return "Both methods are comparable. Choose based on computational resources."
        elif mpnn_avg > llm_avg:
            return "ProteinMPNN sequences show better structural predictions. Recommended for folding accuracy."
        else:
            return "LLM sequences show better properties. May offer more diversity and novelty."
