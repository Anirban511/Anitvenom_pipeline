#!/usr/bin/env python3
"""
================================================================================
TEST SUITE  -  pytest
================================================================================
WHY THIS EXISTS
---------------
Tests are the clearest signal of engineering maturity. They prove the system's
guarantees hold and let you refactor without fear. This suite covers the
infrastructure layer's contracts: config validation, retry semantics, cache
correctness, and input validation.

RUN
---
    pip install pytest
    pytest test_pipeline.py -v

COVERAGE FOCUS
--------------
We test BEHAVIOUR and EDGE CASES, not implementation details:
    - config rejects invalid weights / params  (fail-fast guarantee)
    - retry eventually succeeds AND eventually gives up (both directions)
    - retry does NOT retry on unlisted exceptions (no masking real bugs)
    - cache returns what was stored, and content-addressing is order-stable
    - validators accept valid input and reject every class of bad input
================================================================================
"""

import os
import sys
import pytest

# import the module under test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infrastructure import (  # noqa: E402
    PipelineConfig,
    retry_with_backoff,
    DiskCache,
    validate_pdb_id,
    validate_sequence,
    clean_sequence,
)


# ------------------------------------------------------------------ CONFIG
class TestConfig:
    def test_default_config_is_valid(self):
        cfg = PipelineConfig()
        cfg.validate()  # should not raise
        assert cfg.num_sequences == 5

    def test_weights_must_sum_to_one(self):
        cfg = PipelineConfig(w_structure=0.5, w_stability=0.5, w_docking=0.5)
        with pytest.raises(ValueError, match="sum to 1.0"):
            cfg.validate()

    def test_rejects_negative_sequence_count(self):
        with pytest.raises(ValueError, match="num_sequences"):
            PipelineConfig(num_sequences=0).validate()

    def test_rejects_insane_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            PipelineConfig(temperature=5.0).validate()

    def test_rejects_bad_length_window(self):
        with pytest.raises(ValueError, match="min_length"):
            PipelineConfig(min_length=200, max_length=100).validate()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_NUM_SEQUENCES", "12")
        monkeypatch.setenv("PIPELINE_TEMPERATURE", "0.9")
        cfg = PipelineConfig.from_env()
        assert cfg.num_sequences == 12
        assert cfg.temperature == 0.9


# ------------------------------------------------------------------ RETRY
class TestRetry:
    def test_succeeds_first_try(self):
        calls = {"n": 0}

        @retry_with_backoff(max_retries=3, base_seconds=0.0)
        def ok():
            calls["n"] += 1
            return "done"

        assert ok() == "done"
        assert calls["n"] == 1  # no wasted retries

    def test_recovers_after_transient_failures(self):
        calls = {"n": 0}

        @retry_with_backoff(max_retries=3, base_seconds=0.0)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return "recovered"

        assert flaky() == "recovered"
        assert calls["n"] == 3

    def test_gives_up_after_max_retries(self):
        calls = {"n": 0}

        @retry_with_backoff(max_retries=3, base_seconds=0.0)
        def always_fails():
            calls["n"] += 1
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            always_fails()
        assert calls["n"] == 3  # tried exactly max_retries times

    def test_does_not_retry_unlisted_exceptions(self):
        """A ValueError (our bug) must NOT be retried - fail immediately."""
        calls = {"n": 0}

        @retry_with_backoff(max_retries=3, base_seconds=0.0, exceptions=(ConnectionError,))
        def bug():
            calls["n"] += 1
            raise ValueError("programmer error")

        with pytest.raises(ValueError):
            bug()
        assert calls["n"] == 1  # not retried


# ------------------------------------------------------------------ CACHE
class TestCache:
    def test_set_then_get_roundtrip(self, tmp_path):
        cache = DiskCache(str(tmp_path), enabled=True)
        cache.set("pdb", "3FTX", {"atoms": 100})
        assert cache.get("pdb", "3FTX") == {"atoms": 100}

    def test_miss_returns_none(self, tmp_path):
        cache = DiskCache(str(tmp_path), enabled=True)
        assert cache.get("pdb", "NOPE") is None

    def test_content_addressing_is_order_stable(self, tmp_path):
        """Same logical payload, different dict order -> same cache key."""
        cache = DiskCache(str(tmp_path), enabled=True)
        cache.set("gen", {"a": 1, "b": 2}, "value1")
        assert cache.get("gen", {"b": 2, "a": 1}) == "value1"

    def test_disabled_cache_is_noop(self, tmp_path):
        cache = DiskCache(str(tmp_path), enabled=False)
        cache.set("x", "y", "z")
        assert cache.get("x", "y") is None

    def test_stats_track_hits_and_misses(self, tmp_path):
        cache = DiskCache(str(tmp_path), enabled=True)
        cache.get("a", "1")          # miss
        cache.set("a", "1", "v")
        cache.get("a", "1")          # hit
        s = cache.stats()
        assert s["hits"] == 1 and s["misses"] == 1
        assert s["hit_rate_pct"] == 50.0


# ------------------------------------------------------------------ VALIDATORS
class TestValidators:
    def test_valid_pdb_id_uppercased(self):
        assert validate_pdb_id("3ftx") == "3FTX"

    @pytest.mark.parametrize("bad", ["3FT", "3FTXX", "3F!X", "", "12 4"])
    def test_invalid_pdb_ids_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_pdb_id(bad)

    def test_valid_sequence_passes(self):
        assert validate_sequence("ACDEFGHIK") == "ACDEFGHIK"

    def test_sequence_with_invalid_residue_rejected(self):
        with pytest.raises(ValueError, match="invalid residues"):
            validate_sequence("ACDEFXYZ")  # X, Z not standard

    def test_sequence_length_bounds(self):
        with pytest.raises(ValueError, match="outside"):
            validate_sequence("ACD", min_len=10, max_len=100)

    def test_clean_sequence_strips_invalid(self):
        assert clean_sequence("AC1DE!FG") == "ACDEFG"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
