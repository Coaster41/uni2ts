"""
Tests for plot_probes.py (PR-5).

All tests use MPLBACKEND=Agg (set in conftest or via env) for headless rendering.
No GPU or model checkpoints required.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")

import pytest

from experiments.mech_interp.block1_probing.plot_probes import load_results, load_metadata, plot_probes


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_model_json(directory: str, model_name: str, features: dict[str, dict[str, float]]) -> str:
    """Write a minimal model results JSON and return its path."""
    path = os.path.join(directory, f"{model_name}.json")
    with open(path, "w") as f:
        json.dump(features, f)
    return path


def _write_metadata(directory: str, features: dict) -> str:
    path = os.path.join(directory, "metadata.json")
    with open(path, "w") as f:
        json.dump({"features": features}, f)
    return path


def _tiny_scores(n_layers: int = 2) -> dict[str, float]:
    return {str(i): float(i) * 0.1 for i in range(n_layers)}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoadResults:
    def test_string_layer_keys_converted_to_int(self, tmp_path):
        """JSON layer keys come back as strings; load_results must convert them to ints."""
        data = {"slope": {"0": 0.1, "1": 0.5, "7": 0.9}}
        p = tmp_path / "model.json"
        p.write_text(json.dumps(data))

        result = load_results(str(p))

        assert set(result["slope"].keys()) == {0, 1, 7}
        assert result["slope"][7] == pytest.approx(0.9)

    def test_multiple_features(self, tmp_path):
        data = {
            "slope": {"0": 0.0, "1": 0.3},
            "period_idx": {"0": 0.15, "1": 0.6},
        }
        p = tmp_path / "m.json"
        p.write_text(json.dumps(data))
        result = load_results(str(p))
        assert set(result.keys()) == {"slope", "period_idx"}


class TestLoadMetadata:
    def test_loads_from_file(self, tmp_path):
        meta = {"slope": {"type": "regression", "baseline": 0.0, "metric": "R²"}}
        _write_metadata(str(tmp_path), meta)
        result = load_metadata(str(tmp_path))
        assert result["slope"]["baseline"] == 0.0

    def test_fallback_when_no_metadata_file(self, tmp_path):
        """Should not crash and should return non-empty defaults."""
        result = load_metadata(str(tmp_path))
        assert isinstance(result, dict)
        assert len(result) > 0
        # All known regression features default to baseline 0.0
        for feat in ("slope", "log_noise_var", "phase_cos", "phase_sin"):
            assert result[feat]["baseline"] == pytest.approx(0.0)
        # period_idx baseline is 1/8
        assert result["period_idx"]["baseline"] == pytest.approx(1.0 / 8)


class TestPlotProbes:
    def test_creates_one_pdf_per_feature(self, tmp_path):
        """All features present in both models → one PDF per feature."""
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        results_dir.mkdir()

        features = {
            "slope": _tiny_scores(2),
            "period_idx": _tiny_scores(2),
        }
        _write_model_json(str(results_dir), "model_a", features)
        _write_model_json(str(results_dir), "model_b", features)
        meta = {
            "slope": {"type": "regression", "baseline": 0.0, "metric": "R²"},
            "period_idx": {"type": "classification", "baseline": 0.125, "metric": "accuracy"},
        }
        _write_metadata(str(results_dir), meta)

        plot_probes(str(results_dir), str(figures_dir))

        assert (figures_dir / "probe_slope.pdf").exists()
        assert (figures_dir / "probe_period_idx.pdf").exists()

    def test_missing_feature_in_one_model_skipped(self, tmp_path):
        """If one model is missing a feature, that model's curve is skipped — no crash."""
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        results_dir.mkdir()

        # model_a has slope; model_b is missing slope
        _write_model_json(str(results_dir), "model_a", {"slope": _tiny_scores(2)})
        _write_model_json(str(results_dir), "model_b", {"log_noise_var": _tiny_scores(2)})

        # Should complete without raising, creating PDFs for both features
        plot_probes(str(results_dir), str(figures_dir))

        assert (figures_dir / "probe_slope.pdf").exists()
        assert (figures_dir / "probe_log_noise_var.pdf").exists()

    def test_model_autodiscovery_excludes_metadata(self, tmp_path):
        """metadata.json must not be treated as a model results file."""
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        results_dir.mkdir()

        _write_model_json(str(results_dir), "my_model", {"slope": _tiny_scores(2)})
        _write_metadata(str(results_dir), {"slope": {"type": "regression", "baseline": 0.0, "metric": "R²"}})

        # Should create exactly one plot, not crash on metadata.json as a model
        plot_probes(str(results_dir), str(figures_dir))
        assert (figures_dir / "probe_slope.pdf").exists()
        assert not (figures_dir / "probe_features.pdf").exists()  # metadata key would be "features"

    def test_no_json_files_exits_cleanly(self, tmp_path):
        """Empty results dir → no crash, no figures created."""
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        results_dir.mkdir()

        plot_probes(str(results_dir), str(figures_dir))

        # figures dir may or may not be created, but no PDFs
        if figures_dir.exists():
            assert list(figures_dir.glob("*.pdf")) == []

    def test_metadata_fallback_produces_figures(self, tmp_path):
        """No metadata.json → fallback heuristics → still produces figures without crash."""
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        results_dir.mkdir()

        _write_model_json(str(results_dir), "m", {"slope": _tiny_scores(3), "period_idx": _tiny_scores(3)})
        # Intentionally no metadata.json

        plot_probes(str(results_dir), str(figures_dir))

        assert (figures_dir / "probe_slope.pdf").exists()
        assert (figures_dir / "probe_period_idx.pdf").exists()
