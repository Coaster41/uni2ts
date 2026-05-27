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
from experiments.mech_interp.block1_probing.plot_probes_patch import (
    load_per_patch_results,
    plot_probes_patch,
    _scores_to_grid,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_model_json(directory: str, model_name: str, features: dict[str, dict[str, float]]) -> str:
    """Write a minimal model results JSON and return its path."""
    path = os.path.join(directory, f"{model_name}.json")
    with open(path, "w") as f:
        json.dump(features, f)
    return path


def _write_per_patch_json(directory: str, model_name: str,
                           features: dict, n_layers: int = 2, n_patches: int = 4) -> str:
    """Write a minimal per-patch results JSON with string keys at both levels."""
    data = {
        feat: {
            str(l): {str(p): float(l * 0.1 + p * 0.01) for p in range(n_patches)}
            for l in range(n_layers)
        }
        for feat in features
    }
    path = os.path.join(directory, f"{model_name}_per_patch.json")
    with open(path, "w") as f:
        json.dump(data, f)
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


# ── Per-patch plot tests (PR-5a) ──────────────────────────────────────────────

class TestLoadPerPatchResults:
    def test_nested_string_keys_converted_to_int(self, tmp_path):
        """Layer and patch keys must both be converted from string to int."""
        data = {"slope": {"0": {"0": 0.1, "31": 0.9}, "7": {"0": 0.5, "31": 0.95}}}
        p = tmp_path / "m_per_patch.json"
        p.write_text(json.dumps(data))
        result = load_per_patch_results(str(p))
        assert result["slope"][0][31] == pytest.approx(0.9)
        assert result["slope"][7][0] == pytest.approx(0.5)

    def test_multiple_features(self, tmp_path):
        data = {
            "slope": {"0": {"0": 0.1}},
            "period_idx": {"0": {"0": 0.2}},
        }
        p = tmp_path / "m_per_patch.json"
        p.write_text(json.dumps(data))
        result = load_per_patch_results(str(p))
        assert set(result.keys()) == {"slope", "period_idx"}


class TestScoresToGrid:
    def test_shape(self):
        layer_dict = {0: {0: 0.1, 1: 0.2}, 1: {0: 0.3, 1: 0.4}}
        grid = _scores_to_grid(layer_dict)
        assert grid.shape == (2, 2)
        assert grid[0, 1] == pytest.approx(0.2)
        assert grid[1, 0] == pytest.approx(0.3)


class TestPlotProbesPatch:
    def test_creates_heatmap_and_slice_pdfs(self, tmp_path):
        """Both probe_patch_*.pdf and probe_patch_slice_*.pdf must be created per feature."""
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        results_dir.mkdir()

        for model in ("model_a", "model_b"):
            _write_per_patch_json(str(results_dir), model, ["slope", "period_idx"])

        plot_probes_patch(str(results_dir), str(figures_dir))

        assert (figures_dir / "probe_patch_slope.pdf").exists()
        assert (figures_dir / "probe_patch_period_idx.pdf").exists()
        assert (figures_dir / "probe_patch_slice_slope.pdf").exists()
        assert (figures_dir / "probe_patch_slice_period_idx.pdf").exists()

    def test_missing_feature_in_one_model_no_crash(self, tmp_path):
        """One model missing a feature → skip that model's contribution, no crash."""
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        results_dir.mkdir()

        _write_per_patch_json(str(results_dir), "model_a", ["slope"])
        _write_per_patch_json(str(results_dir), "model_b", ["log_noise_var"])

        plot_probes_patch(str(results_dir), str(figures_dir))

        assert (figures_dir / "probe_patch_slope.pdf").exists()
        assert (figures_dir / "probe_patch_log_noise_var.pdf").exists()

    def test_no_per_patch_files_exits_cleanly(self, tmp_path):
        """No *_per_patch.json → no crash, no figures created."""
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        results_dir.mkdir()
        # Only a regular (non-per-patch) model json
        _write_model_json(str(results_dir), "model_a", {"slope": _tiny_scores(2)})

        plot_probes_patch(str(results_dir), str(figures_dir))

        if figures_dir.exists():
            assert list(figures_dir.glob("probe_patch_*.pdf")) == []

    def test_metadata_excluded_from_model_discovery(self, tmp_path):
        """metadata.json must not be picked up as a per-patch model file."""
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures"
        results_dir.mkdir()

        _write_per_patch_json(str(results_dir), "my_model", ["slope"])
        _write_metadata(str(results_dir), {"slope": {"type": "regression", "baseline": 0.0, "metric": "R²"}})

        plot_probes_patch(str(results_dir), str(figures_dir))
        assert (figures_dir / "probe_patch_slope.pdf").exists()
        # Should not create a file for "metadata" as a model name
        assert not (figures_dir / "probe_patch_features.pdf").exists()
