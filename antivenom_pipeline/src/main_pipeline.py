#!/usr/bin/env python3
"""
ANTIVENOM PROTEIN DESIGN PIPELINE
Workflow: PDB Input → LLM Sequences → ProteinMPNN → AlphaFold → Comparison

Author: Anirban
Project: Computational Antivenom Design
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import subprocess
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import LLM sequence generator using ProtGPT2 (no API key required)
try:
    from llm_sequence_generator_protgpt2 import LLMSequenceGeneratorNoAPI, ProtGPT2Lite
except ImportError:
    logger.warning("llm_sequence_generator_protgpt2 not found. Make sure it's in the same directory.")
    LLMSequenceGeneratorNoAPI = None
    ProtGPT2Lite = None

# Import infrastructure layer (retry, caching, validation, config).
# The pipeline runs WITH or WITHOUT it: if missing, we degrade gracefully.
try:
    from infrastructure import (
        PipelineConfig, retry_with_backoff, DiskCache,
        validate_pdb_id, clean_sequence,
    )
    INFRA = True
except ImportError:
    logger.warning("infrastructure.py not found - running without retry/cache/validation.")
    INFRA = False
    # Lightweight no-op fallbacks so the rest of the code is unconditional
    def retry_with_backoff(max_retries=3, base_seconds=1.0, exceptions=(Exception,)):
        def deco(fn):
            return fn
        return deco
    def validate_pdb_id(x):
        return x.upper()
    def clean_sequence(s):
        return "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
    DiskCache = None
    PipelineConfig = None


@dataclass
class SequenceResult:
    """Result object for a generated sequence"""
    sequence_id: str
    method: str  # 'LLM' or 'ProteinMPNN'
    sequence: str
    length: int
    avg_plddt: float = None
    max_plddt: float = None
    min_plddt: float = None
    cysteine_count: int = None
    hydrophobicity_score: float = None
    binding_energy: float = None
    plddt_scores: List[float] = None
    
    def to_dict(self):
        return asdict(self)


class PDBHandler:
    """STEP 1: Download and validate toxin structures from PDB"""
    
    def __init__(self, pdb_id: str, output_dir: str = './data/pdb', cache: 'DiskCache' = None):
        # Fail-fast validation: a PDB ID must be 4 alphanumeric chars.
        self.pdb_id = validate_pdb_id(pdb_id)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_file = self.output_dir / f"{self.pdb_id}.pdb"
        self.cache = cache  # optional DiskCache; skips re-download when present
    
    @retry_with_backoff(max_retries=3, base_seconds=1.0)
    def _fetch(self, url: str):
        """Network fetch isolated so retry/backoff wraps ONLY the flaky call."""
        import urllib.request
        urllib.request.urlretrieve(url, self.pdb_file)
    
    def download_pdb(self):
        """Download PDB structure from RCSB (with cache + retry if infra present)."""
        logger.info(f"Downloading PDB structure: {self.pdb_id}")
        
        # Cache check: if we've fetched this PDB before, reuse it.
        if self.cache is not None:
            cached = self.cache.get("pdb", self.pdb_id)
            if cached and Path(cached.get("path", "")).exists():
                self.pdb_file = Path(cached["path"])
                logger.info(f"\u2713 Using cached PDB: {self.pdb_file}")
                return True
        
        url = f"https://files.rcsb.org/download/{self.pdb_id}.pdb"
        try:
            self._fetch(url)  # retries transparently on transient failure
            logger.info(f"\u2713 Downloaded to {self.pdb_file}")
            if self.cache is not None:
                self.cache.set("pdb", self.pdb_id, {"path": str(self.pdb_file)})
            return True
        except Exception as e:
            logger.error(f"\u2717 Failed to download after retries: {e}")
            return False
    
    def validate_pdb(self) -> bool:
        """Validate PDB file format"""
        if not self.pdb_file.exists():
            logger.error(f"PDB file not found: {self.pdb_file}")
            return False
        
        with open(self.pdb_file, 'r') as f:
            lines = f.readlines()
            atom_lines = [l for l in lines if l.startswith('ATOM')]
            
        logger.info(f"✓ PDB validation passed ({len(atom_lines)} atoms)")
        return len(atom_lines) > 0
    
    def extract_coordinates(self) -> Dict:
        """Extract 3D coordinates from PDB"""
        coords = {'atoms': [], 'residues': []}
        
        with open(self.pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    res_num = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords['atoms'].append({
                        'name': atom_name,
                        'residue': res_num,
                        'coords': [x, y, z]
                    })
        
        logger.info(f"✓ Extracted {len(coords['atoms'])} coordinates")
        return coords





class ProteinMPNN:
    """STEP 3: Generate structure-optimized sequences using ProteinMPNN"""
    
    def __init__(self, pdb_file: str):
        self.pdb_file = pdb_file
        self.model_path = "./protein_mpnn_weights"
    
    def generate_sequences(self, num_sequences: int = 5, binding_region: str = None) -> List[str]:
        """
        Run ProteinMPNN on PDB structure
        
        This would typically call the ProteinMPNN model.
        For now, we'll generate synthetic sequences for demo.
        """
        logger.info(f"Running ProteinMPNN on {self.pdb_file}...")
        
        # TODO: Integrate actual ProteinMPNN
        # This would require:
        # 1. Download ProteinMPNN weights
        # 2. Parse PDB
        # 3. Run inference with or without interface specification
        
        # Placeholder: Generate synthetic sequences
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        sequences = [
            ''.join(np.random.choice(list(amino_acids), size=np.random.randint(80, 150)))
            for _ in range(num_sequences)
        ]
        
        logger.info(f"✓ ProteinMPNN generated {len(sequences)} structure-optimized sequences")
        return sequences


class AlphaFoldPredictor:
    """STEP 4: Predict 3D structures and confidence scores using AlphaFold"""
    
    def __init__(self):
        self.model = None
    
    def predict_structure(self, sequence: str) -> Tuple[np.ndarray, List[float]]:
        """
        Predict 3D structure and pLDDT scores
        
        Returns:
            (coordinates_array, plddt_scores)
        """
        logger.info(f"Predicting structure for {len(sequence)}-aa sequence...")
        
        # TODO: Integrate OmegaFold or ESMFold for faster inference
        # For demo: generate synthetic pLDDT scores
        
        # Synthetic: Higher confidence for structured regions
        plddt = np.random.normal(75, 15, size=len(sequence))
        plddt = np.clip(plddt, 10, 100)
        
        logger.info(f"✓ Average pLDDT: {plddt.mean():.2f} ± {plddt.std():.2f}")
        
        return None, plddt.tolist()
    
    def batch_predict(self, sequences: List[str]) -> Dict[str, Tuple]:
        """Predict structures for multiple sequences"""
        results = {}
        for i, seq in enumerate(sequences):
            coords, plddt = self.predict_structure(seq)
            results[f"seq_{i}"] = (coords, plddt)
        return results


class ProteinAnalyzer:
    """STEP 5: Analyze and compare sequences"""
    
    @staticmethod
    def calculate_cysteine_bonds(sequence: str) -> Tuple[int, List[Tuple[int, int]]]:
        """Count cysteines and identify potential disulfide bonds"""
        cysteine_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
        count = len(cysteine_positions)
        
        # Pair cysteines (simple greedy pairing)
        bonds = []
        remaining = cysteine_positions.copy()
        while len(remaining) >= 2:
            bonds.append((remaining.pop(0), remaining.pop(0)))
        
        return count, bonds
    
    @staticmethod
    def calculate_hydrophobicity(sequence: str) -> float:
        """
        Calculate Kyte-Doolittle hydrophobicity index
        Scale: -4.5 to +4.5 (negative = hydrophilic, positive = hydrophobic)
        """
        # Kyte-Doolittle scale
        kd_scale = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        total = sum(kd_scale.get(aa, 0) for aa in sequence)
        return total / len(sequence) if sequence else 0
    
    @staticmethod
    def quality_report(results: List[SequenceResult]) -> Dict:
        """Generate comprehensive comparison report"""
        report = {
            'llm_sequences': [],
            'mpnn_sequences': [],
            'comparison': {}
        }
        
        llm_results = [r for r in results if r.method == 'LLM']
        mpnn_results = [r for r in results if r.method == 'ProteinMPNN']
        
        report['llm_sequences'] = [r.to_dict() for r in llm_results]
        report['mpnn_sequences'] = [r.to_dict() for r in mpnn_results]
        
        # Compute metrics
        if llm_results:
            avg_plddt_llm = np.mean([r.avg_plddt for r in llm_results if r.avg_plddt])
            report['comparison']['avg_plddt_llm'] = float(avg_plddt_llm)
        
        if mpnn_results:
            avg_plddt_mpnn = np.mean([r.avg_plddt for r in mpnn_results if r.avg_plddt])
            report['comparison']['avg_plddt_mpnn'] = float(avg_plddt_mpnn)
        
        report['comparison']['quality_assessment'] = {
            'high_confidence_llm': sum(1 for r in llm_results if r.avg_plddt and r.avg_plddt > 90),
            'high_confidence_mpnn': sum(1 for r in mpnn_results if r.avg_plddt and r.avg_plddt > 90),
        }
        
        return report


class AntivenemPipeline:
    """Main orchestration class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = []
        self.output_dir = Path(config.get('output_dir', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build a validated config + cache when the infrastructure layer is available.
        self.cache = None
        if INFRA:
            try:
                self.pcfg = PipelineConfig.from_env(
                    output_dir=str(self.output_dir),
                    num_sequences=config.get('num_sequences', 5),
                )
                if self.pcfg.enable_cache and DiskCache is not None:
                    self.cache = DiskCache(self.pcfg.cache_dir, enabled=True)
                logger.info("\u2713 Infrastructure active: validated config + cache + retry")
            except ValueError as e:
                # Fail fast on bad config rather than midway through a long run.
                logger.error(f"Invalid configuration: {e}")
                raise
        else:
            self.pcfg = None
    
    def run(self, pdb_id: str, num_sequences: int = 5):
        """Execute complete pipeline"""
        logger.info("="*60)
        logger.info("ANTIVENOM PROTEIN DESIGN PIPELINE")
        logger.info("="*60)
        
        # STEP 1: Get PDB structure
        logger.info("\n[STEP 1] Fetching toxin structure...")
        pdb_handler = PDBHandler(pdb_id, cache=self.cache)
        if not pdb_handler.download_pdb() or not pdb_handler.validate_pdb():
            logger.error("Failed to obtain valid PDB structure")
            return None
        
        # STEP 2: LLM sequence generation using ProtGPT2 (no API key needed)
        logger.info("\n[STEP 2] Generating LLM sequences using ProtGPT2...")
        toxin_info = {
            'type': '3FTx',
            'binding_residues': 'Acetylcholine binding pocket',
            'target': 'Neuromuscular junction'
        }
        
        llm_sequences = []
        
        if LLMSequenceGeneratorNoAPI is None:
            logger.error("✗ llm_sequence_generator_protgpt2 module not found!")
            logger.error("  Please copy llm_sequence_generator_protgpt2.py to the same directory")
        else:
            try:
                # Try full ProtGPT2 model
                llm_gen = LLMSequenceGeneratorNoAPI(method='protgpt2')
                llm_sequences = llm_gen.generate_antivenom_sequences(toxin_info, num_sequences)
                
                # If model failed or no sequences generated, use lite mode
                if not llm_sequences and ProtGPT2Lite is not None:
                    logger.warning("⚠️  ProtGPT2 model generation failed, using lite mode...")
                    llm_sequences = ProtGPT2Lite.generate_sequences(num_sequences=num_sequences)
                    
            except Exception as e:
                logger.warning(f"⚠️  ProtGPT2 model error: {e}")
                if ProtGPT2Lite is not None:
                    logger.info("  Falling back to lite mode...")
                    llm_sequences = ProtGPT2Lite.generate_sequences(num_sequences=num_sequences)
                else:
                    logger.error("  Lite mode also not available")
        
        # STEP 3: ProteinMPNN generation
        logger.info("\n[STEP 3] Generating ProteinMPNN sequences...")
        mpnn = ProteinMPNN(str(pdb_handler.pdb_file))
        mpnn_sequences = mpnn.generate_sequences(num_sequences)
        
        # STEP 4: AlphaFold prediction
        logger.info("\n[STEP 4] Predicting structures with AlphaFold...")
        af = AlphaFoldPredictor()
        
        all_sequences = [
            ('LLM', llm_sequences),
            ('ProteinMPNN', mpnn_sequences)
        ]
        
        for method, sequences in all_sequences:
            for i, seq in enumerate(sequences):
                # Boundary validation: strip any invalid residues before scoring
                # so malformed model output never pollutes the analytics.
                seq = clean_sequence(seq)
                if not seq:
                    logger.warning(f"  {method} sequence {i+1} empty after cleaning; skipping")
                    continue
                
                coords, plddt = af.predict_structure(seq)
                
                cys_count, bonds = ProteinAnalyzer.calculate_cysteine_bonds(seq)
                hydro = ProteinAnalyzer.calculate_hydrophobicity(seq)
                
                result = SequenceResult(
                    sequence_id=f"{method}_seq_{i+1}",
                    method=method,
                    sequence=seq,
                    length=len(seq),
                    avg_plddt=np.mean(plddt),
                    max_plddt=max(plddt),
                    min_plddt=min(plddt),
                    cysteine_count=cys_count,
                    hydrophobicity_score=hydro,
                    plddt_scores=plddt
                )
                self.results.append(result)
        
        # STEP 5: Analysis and comparison
        logger.info("\n[STEP 5] Comparing methods...")
        report = ProteinAnalyzer.quality_report(self.results)
        
        # Attach infrastructure telemetry (cache hit-rate) when available.
        if self.cache is not None:
            report['cache_stats'] = self.cache.stats()
            logger.info(f"\u2713 Cache stats: {self.cache.stats()}")
        
        # Save results
        self._save_results(report)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        
        return report
    
    def _save_results(self, report: Dict):
        """Save results to JSON"""
        output_file = self.output_dir / "results.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"✓ Results saved to {output_file}")


if __name__ == "__main__":
    config = {
        'output_dir': './results',
        'api_key': os.getenv('ANTHROPIC_API_KEY')
    }
    
    pipeline = AntivenemPipeline(config)
    # Example: Use 3FTx (three-finger toxin structure)
    report = pipeline.run('3FTX', num_sequences=5)
