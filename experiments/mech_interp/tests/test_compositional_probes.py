import math

import numpy as np
import pytest

from uni2ts.model.moiraie.module import MoiraieModule

from experiments.mech_interp.lib.synthetic import generate_composite_dataset
from experiments.mech_interp.block1_probing.run_compositional_probes import (
    COMPOSITIONAL_CONCEPTS,
    run_compositional_probes,
)

_TINY = dict(
    d_model=64,
    d_ff=128,
    num_layers=2,
    patch_size=16,
    max_seq_len=64,
    attn_dropout_p=0.0,
    dropout_p=0.0,
)


@pytest.fixture
def module_e():
    return MoiraieModule(**_TINY, num_predict_token=1).eval()


def test_probe_transfer_pipeline(module_e):
    ds = generate_composite_dataset(n=500, seed=0)
    idx = np.random.default_rng(0).permutation(500)
    results = run_compositional_probes(
        module_e, ds, idx[:400], idx[400:],
        batch_size=8, pooling="mean_ctx",
    )
    assert len(results) > 0
    expected_layers = {-1} | set(range(_TINY["num_layers"]))
    for c1_name, inner in results.items():
        for c2_name, layer_dict in inner.items():
            assert set(layer_dict.keys()) == expected_layers, \
                f"{c1_name}→{c2_name}: unexpected layer keys {set(layer_dict.keys())}"
            for layer_idx, scores in layer_dict.items():
                assert "atomic" in scores
                assert math.isfinite(scores["atomic"]), \
                    f"{c1_name}→{c2_name} L{layer_idx}: atomic not finite"
                if scores["delta"] is not None:
                    assert math.isfinite(scores["delta"]), \
                        f"{c1_name}→{c2_name} L{layer_idx}: delta not finite"


def test_pair_filtering():
    concept_mask = np.zeros((10, 5), dtype=bool)
    # rows 0-2: only concept 0 (atomic trend)
    concept_mask[:3, 0] = True
    # rows 3-5: concepts 0 + 1 (pair trend+level)
    concept_mask[3:6, 0] = True
    concept_mask[3:6, 1] = True
    # rows 6-9: concept 1 only
    concept_mask[6:, 1] = True

    n_concepts = concept_mask.sum(axis=1)
    atomic_0 = (n_concepts == 1) & concept_mask[:, 0]
    pair_01   = (n_concepts == 2) & concept_mask[:, 0] & concept_mask[:, 1]
    atomic_1  = (n_concepts == 1) & concept_mask[:, 1]

    assert atomic_0.sum() == 3
    assert pair_01.sum()   == 3
    assert atomic_1.sum()  == 4


def test_output_schema(module_e):
    ds = generate_composite_dataset(n=500, seed=1)
    idx = np.random.default_rng(1).permutation(500)
    results = run_compositional_probes(
        module_e, ds, idx[:400], idx[400:],
        batch_size=8, pooling="mean_ctx",
    )
    concept_names = [c[0] for c in COMPOSITIONAL_CONCEPTS]
    for c1_name, inner in results.items():
        assert c1_name in concept_names
        for c2_name, layer_dict in inner.items():
            assert c2_name != c1_name
            for layer_idx, entry in layer_dict.items():
                assert set(entry.keys()) == {"atomic", "pair", "delta"}, \
                    f"Missing keys in {c1_name}→{c2_name} L{layer_idx}: {entry.keys()}"
