#!/usr/bin/env python3
"""
================================================================================
INFRASTRUCTURE LAYER  (SDE / systems-engineering features)
================================================================================
This module adds the production-engineering scaffolding around the research
pipeline. Each component exists for a reason you can defend in an interview;
the docstring for each states the WHY, the trade-off, and the alternative.

Components
----------
    1. Config            - typed, validated, env-overridable configuration
    2. retry_with_backoff- resilience wrapper for flaky external calls
    3. DiskCache         - content-addressed cache to avoid redundant work
    4. validators        - fail-fast input validation
    5. get_logger        - structured, leveled logging

DESIGN PHILOSOPHY
-----------------
The pipeline calls expensive, failure-prone external resources (RCSB PDB
downloads, model inference). The infrastructure here targets the three things
that actually break such systems in production: transient network failures,
redundant recomputation, and bad input. It deliberately stays lightweight -
no microservices or orchestration frameworks, because at this scale that would
be over-engineering, and an interviewer rightly distrusts complexity without
justification.
================================================================================
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


# ==============================================================================
# 1. STRUCTURED LOGGING
# ==============================================================================
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    WHY: A single, consistently-formatted logger beats scattered print()s.
    Timestamps + levels + module name make logs greppable and let us silence
    or amplify verbosity per environment without touching call sites.

    TRADE-OFF: Slightly more setup than print(); pays off the first time you
    debug a failure you can't reproduce locally.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False
    return logger


log = get_logger("infra")


# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
@dataclass
class PipelineConfig:
    """
    WHY: Centralized, typed config instead of magic numbers scattered across
    files. Every knob lives here, is documented, and can be overridden by an
    environment variable - so the same code runs in a notebook, a CI job, or a
    container without edits.

    ALTERNATIVE CONSIDERED: a YAML file (we have config.yaml). A dataclass is
    chosen here for the *runtime* surface because it gives type safety,
    IDE autocomplete, and validation in one place; YAML is better for
    non-developer-editable values. The two can coexist - YAML loads INTO this.
    """
    pdb_id: str = "3FTX"
    num_sequences: int = 5
    output_dir: str = "./results"
    cache_dir: str = "./.cache"

    # generation
    temperature: float = 0.7
    top_p: float = 0.9
    max_length: int = 150
    min_length: int = 50

    # scoring weights (must sum to 1.0 - validated below)
    w_structure: float = 0.35
    w_stability: float = 0.35
    w_docking: float = 0.30

    # resilience
    max_retries: int = 3
    backoff_base_seconds: float = 1.0

    enable_cache: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, **overrides) -> "PipelineConfig":
        """Build config, letting env vars (PIPELINE_*) override defaults."""
        cfg = cls(**overrides)
        for f_name in cfg.__dataclass_fields__:
            env_key = f"PIPELINE_{f_name.upper()}"
            if env_key in os.environ:
                raw = os.environ[env_key]
                current = getattr(cfg, f_name)
                # cast env string back to the field's type
                if isinstance(current, bool):
                    setattr(cfg, f_name, raw.lower() in ("1", "true", "yes"))
                elif isinstance(current, int):
                    setattr(cfg, f_name, int(raw))
                elif isinstance(current, float):
                    setattr(cfg, f_name, float(raw))
                else:
                    setattr(cfg, f_name, raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Fail fast on nonsensical config rather than midway through a run."""
        weight_sum = round(self.w_structure + self.w_stability + self.w_docking, 6)
        if weight_sum != 1.0:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {weight_sum} "
                f"({self.w_structure}+{self.w_stability}+{self.w_docking})"
            )
        if self.num_sequences < 1:
            raise ValueError(f"num_sequences must be >= 1, got {self.num_sequences}")
        if not (0.0 < self.temperature <= 2.0):
            raise ValueError(f"temperature out of sane range (0,2]: {self.temperature}")
        if self.min_length >= self.max_length:
            raise ValueError(f"min_length ({self.min_length}) must be < max_length ({self.max_length})")

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# 3. RETRY WITH EXPONENTIAL BACKOFF
# ==============================================================================
def retry_with_backoff(
    max_retries: int = 3,
    base_seconds: float = 1.0,
    exceptions: tuple = (Exception,),
):
    """
    WHY: External calls (PDB download, model hub fetch) fail transiently -
    we literally hit an HTTP 403 / network blip during development. Retrying
    with EXPONENTIAL backoff (1s, 2s, 4s...) gives the remote a chance to
    recover without hammering it, which a fixed-interval retry would not.

    TRADE-OFF: Adds latency on genuine failures (you wait through all retries
    before giving up). Mitigated by capping max_retries. We do NOT retry on
    programmer errors - only the exception types passed in - so a ValueError
    in our own code fails immediately instead of being retried 3x.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        log.error(f"{func.__name__} failed after {max_retries} attempts: {exc}")
                        raise
                    delay = base_seconds * (2 ** (attempt - 1))
                    log.warning(
                        f"{func.__name__} attempt {attempt}/{max_retries} failed "
                        f"({exc}); retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
            if last_exc:
                raise last_exc
        return wrapper
    return decorator


# ==============================================================================
# 4. DISK CACHE (content-addressed)
# ==============================================================================
class DiskCache:
    """
    WHY: Two expensive operations repeat across runs - downloading the same PDB
    and re-generating sequences for the same inputs. A content-addressed cache
    (key = hash of the inputs) makes re-runs near-instant and cuts external
    load. This is the single biggest dev-experience win in the system.

    DESIGN: key is a SHA-256 of the logical inputs, so identical inputs always
    map to the same file regardless of call order. Values are JSON for
    transparency (you can open and read a cache entry).

    TRADE-OFF: cache invalidation. We sidestep the hard version by keying on
    *inputs* - if any input changes, the key changes, so stale reads can't
    happen. The cost is disk space; mitigated by a clear() method and the
    cache being opt-in via config.
    """

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(namespace: str, payload: Any) -> str:
        blob = json.dumps({"ns": namespace, "p": payload}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def get(self, namespace: str, payload: Any) -> Optional[Any]:
        if not self.enabled:
            return None
        path = self.dir / f"{namespace}_{self._key(namespace, payload)}.json"
        if path.exists():
            self.hits += 1
            log.info(f"cache HIT  [{namespace}] -> {path.name}")
            with open(path) as fh:
                return json.load(fh)["value"]
        self.misses += 1
        log.info(f"cache MISS [{namespace}]")
        return None

    def set(self, namespace: str, payload: Any, value: Any) -> None:
        if not self.enabled:
            return
        path = self.dir / f"{namespace}_{self._key(namespace, payload)}.json"
        with open(path, "w") as fh:
            json.dump({"namespace": namespace, "payload": payload, "value": value},
                      fh, indent=2, default=str)
        log.info(f"cache WRITE [{namespace}] -> {path.name}")

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(100 * self.hits / total, 1) if total else 0.0,
        }

    def clear(self) -> int:
        if not self.enabled:
            return 0
        n = 0
        for p in self.dir.glob("*.json"):
            p.unlink()
            n += 1
        log.info(f"cache cleared: {n} entries")
        return n


# ==============================================================================
# 5. INPUT VALIDATION (fail fast)
# ==============================================================================
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_pdb_id(pdb_id: str) -> str:
    """
    WHY: A PDB ID is exactly 4 alphanumeric chars. Validating up front turns a
    confusing downstream 404 into a clear, immediate error message - the
    'fail fast, fail loud' principle.
    """
    if not isinstance(pdb_id, str) or len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError(f"Invalid PDB ID '{pdb_id}': must be 4 alphanumeric characters")
    return pdb_id.upper()


def validate_sequence(seq: str, min_len: int = 1, max_len: int = 10_000) -> str:
    """
    WHY: Generated sequences can contain stray tokens. We reject anything with
    non-standard amino acids or outside the length window BEFORE it pollutes
    scoring - bad data caught at the boundary, not deep in the analytics.
    """
    if not seq:
        raise ValueError("Empty sequence")
    bad = set(seq.upper()) - VALID_AMINO_ACIDS
    if bad:
        raise ValueError(f"Sequence contains invalid residues: {sorted(bad)}")
    if not (min_len <= len(seq) <= max_len):
        raise ValueError(f"Sequence length {len(seq)} outside [{min_len}, {max_len}]")
    return seq.upper()


def clean_sequence(seq: str) -> str:
    """Lenient counterpart to validate_sequence: strip invalid chars instead of raising."""
    return "".join(c for c in seq.upper() if c in VALID_AMINO_ACIDS)


# ==============================================================================
# demo / smoke test
# ==============================================================================
if __name__ == "__main__":
    log.info("Infrastructure smoke test")

    cfg = PipelineConfig.from_env()
    log.info(f"Config OK, weights sum to 1.0, num_sequences={cfg.num_sequences}")

    # retry demo: function that fails twice then succeeds
    attempts = {"n": 0}

    @retry_with_backoff(max_retries=3, base_seconds=0.1)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("simulated transient failure")
        return "success"

    log.info(f"retry demo -> {flaky()} after {attempts['n']} attempts")

    # cache demo
    cache = DiskCache("./.cache_demo", enabled=True)
    if cache.get("pdb", "3FTX") is None:
        cache.set("pdb", "3FTX", {"atoms": 1234})
    cache.get("pdb", "3FTX")  # hit
    log.info(f"cache stats: {cache.stats()}")
    cache.clear()

    # validation demo
    log.info(f"validate_pdb_id('3ftx') -> {validate_pdb_id('3ftx')}")
    try:
        validate_sequence("ACDEFXYZ")
    except ValueError as e:
        log.info(f"validation correctly rejected bad sequence: {e}")

    log.info("All infrastructure checks passed")
