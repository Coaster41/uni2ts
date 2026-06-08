"""
Adapter for Toto 2.0 (DataDog/toto2, e.g. ``Datadog/Toto-2.0-22m``).

Toto 2.0 is quantile-head based (not sampling), and its ``forecast()`` already
emits the canonical decile grid, so this maps straight onto our ``[n, 9, H]``
contract with no sample reduction.

API (from the Toto 2.0 model card)::

    from toto2 import Toto2Model
    model = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m").to(device).eval()

    target      = torch.randn(batch, n_variates, time)   # (B, V, T)
    target_mask = torch.ones_like(target, dtype=torch.bool)
    series_ids  = torch.zeros(batch, n_variates, dtype=torch.long)

    quantiles = model.forecast(
        {"target": target, "target_mask": target_mask, "series_ids": series_ids},
        horizon=H, decode_block_size=768, has_missing_values=False,
    )                                  # (9, batch, n_variates, H), levels 0.1..0.9

We feed each stress series as one univariate batch element (V=1), so the output
is ``(9, b, 1, H)`` -> squeeze V -> ``(b, 9, H)``.

Requires the ``toto2`` package (https://github.com/DataDog/toto) which is *not*
installed in the uni2ts venv — run in an env that has it.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastAdapter, batcher


class TotoAdapter(ForecastAdapter):
    def __init__(
        self,
        model_id: str = "Datadog/Toto-2.0-22m",
        device: str = "cuda",
        decode_block_size: int = 768,
        **_ignored,
    ):
        import torch
        from toto2 import Toto2Model

        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = Toto2Model.from_pretrained(model_id).to(self.device).eval()
        self.decode_block_size = decode_block_size

    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        batch_size: int = 64,
    ) -> np.ndarray:
        import torch

        context = np.asarray(context, dtype=np.float32)
        out: list[np.ndarray] = []
        for chunk in batcher(list(context), batch_size):
            arr = np.stack(chunk, axis=0)  # [b, ctx_len]
            b = arr.shape[0]
            # (batch, n_variates=1, time)
            target = torch.tensor(
                arr, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            target_mask = torch.ones_like(target, dtype=torch.bool)
            series_ids = torch.zeros(b, 1, dtype=torch.long, device=self.device)
            with torch.no_grad():
                q = self.model.forecast(
                    {
                        "target": target,
                        "target_mask": target_mask,
                        "series_ids": series_ids,
                    },
                    horizon=horizon,
                    decode_block_size=self.decode_block_size,
                    has_missing_values=False,
                )
            q = np.asarray(q.detach().cpu().float().numpy())  # [9, b, 1, H]
            q = q[:, :, 0, :]  # squeeze n_variates -> [9, b, H]
            out.append(np.moveaxis(q, 0, 1)[:, :, :horizon].astype(np.float32))
        return np.concatenate(out, axis=0).astype(np.float32)  # [n, 9, H]
