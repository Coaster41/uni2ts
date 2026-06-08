"""
Adapter for TimesFM 2.5 (torch).

Mirrors the gift-eval ``timesfm2p5`` notebook. In this build the 2.5 torch model
class lives inside the ``timesfm_2p5_torch`` submodule and is constructed with no
args + ``load_checkpoint()`` (there is no ``from_pretrained``); the forecast
config comes from ``timesfm.configs.ForecastConfig`` and is applied via
``compile()``::

    from timesfm import configs
    from timesfm.timesfm_2p5 import timesfm_2p5_torch
    cls = timesfm_2p5_torch.TimesFM_2p5_200M_torch
    tfm = cls()
    weights = hf_hub_download(cls.DEFAULT_REPO_ID, cls.WEIGHTS_FILENAME)
    tfm.model.load_checkpoint(weights)        # loads weights + moves to cuda:0
    tfm.compile(forecast_config=configs.ForecastConfig(...))
    _, full_preds = tfm.forecast(horizon=H, inputs=context_list)

(We construct it manually instead of via from_pretrained because newer
huggingface_hub leaks `proxies`/`resume_download` into the constructor.)

``tfm.forecast`` returns ``(point_forecast, full_preds)`` where
``full_preds[..., 0]`` is the point forecast and ``full_preds[..., 1:]`` are the
9 quantile columns (deciles). We slice those and transpose to ``[b, Q, H]``.

Requires the ``timesfm`` package (2.5) which is *not* installed in the uni2ts
venv — run in an env that has it.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastAdapter, batcher


class TimesFM25Adapter(ForecastAdapter):
    def __init__(
        self,
        checkpoint: str | None = None,
        per_core_batch_size: int | None = None,
        max_context_cap: int = 4096,
        max_horizon: int | None = None,
        device: str = "auto",
        tfm=None,
        **_ignored,
    ):
        # `None` => auto-size to the actual workload at compile time so the
        # compiled buffers (and thus VRAM) match what we actually forecast.
        self.per_core_batch_size = per_core_batch_size
        self.max_context_cap = max_context_cap
        self.max_horizon = max_horizon
        self._compiled_key = None
        if tfm is not None:
            # Allow injecting a pre-built (already checkpoint-loaded) model.
            self.tfm = tfm
            self._move_to_device(device)
            return
        import os

        from huggingface_hub import hf_hub_download
        from timesfm.timesfm_2p5 import timesfm_2p5_torch

        # We replicate what TimesFM_2p5_200M_torch._from_pretrained does, rather
        # than calling .from_pretrained() directly: newer huggingface_hub passes
        # `proxies`/`resume_download` through to the constructor, which this
        # timesfm build doesn't accept (TypeError). The real loading is
        # `self.model.load_checkpoint(file)` — the wrapper itself doesn't define
        # load_checkpoint — and that nn.Module method does `self.to(self.device)`
        # (cuda:0 if available), which fixes the cpu/cuda mismatch.
        cls = timesfm_2p5_torch.TimesFM_2p5_200M_torch
        repo = checkpoint or cls.DEFAULT_REPO_ID
        self.tfm = cls()
        if os.path.isdir(repo):
            weights = os.path.join(repo, cls.WEIGHTS_FILENAME)
        else:
            weights = hf_hub_download(repo_id=repo, filename=cls.WEIGHTS_FILENAME)
        self.tfm.model.load_checkpoint(weights, torch_compile=self.tfm.torch_compile)
        self._move_to_device(device)

    def _move_to_device(self, device: str) -> None:
        """Move the underlying module to `device` and sync `model.device`, the
        attribute `forecast()` uses to place its inputs. timesfm defaults to
        cuda:0; pass e.g. 'cuda:7' or 'cpu' to override. 'auto'/None keeps
        timesfm's own choice (cuda:0 if available)."""
        if device in (None, "auto"):
            return
        import torch

        dev = torch.device(device)
        self.tfm.model.to(dev)
        self.tfm.model.device = dev

    def _ensure_compiled(self, ctx_len: int, horizon: int, batch_size: int) -> None:
        """(Re)compile sized to the actual workload to keep VRAM bounded.
        ``max_context``/``max_horizon`` are rounded up to a multiple of the model
        patch length ``tfm.model.p`` (not inflated to a fixed 1024 horizon), and
        ``per_core_batch_size`` defaults to the forecast batch size so buffers
        aren't allocated larger than what we feed."""
        from timesfm import configs

        p = self.tfm.model.p
        round_up = lambda x: max(p, ((x + p - 1) // p) * p)

        max_context = min(self.max_context_cap, round_up(ctx_len))
        max_horizon = round_up(self.max_horizon if self.max_horizon else horizon)
        pcbs = self.per_core_batch_size or min(batch_size, 64)
        key = (max_context, max_horizon, pcbs)
        if self._compiled_key == key:
            return
        self.tfm.compile(
            forecast_config=configs.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                infer_is_positive=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                force_flip_invariance=True,
                return_backcast=False,
                normalize_inputs=True,
                per_core_batch_size=pcbs,
            )
        )
        self._compiled_key = key

    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        batch_size: int = 64,
    ) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        self._ensure_compiled(context.shape[1], horizon, batch_size)
        n_q = len(self.quantile_levels)
        out: list[np.ndarray] = []
        for chunk in batcher(list(context), batch_size):
            inputs = [row.astype(np.float32) for row in chunk]
            _, full_preds = self.tfm.forecast(horizon=horizon, inputs=inputs)
            full_preds = np.asarray(full_preds, dtype=np.float32)
            # [b, H, 1 + Q] -> quantile cols -> [b, Q, H]
            q = full_preds[:, :horizon, 1 : 1 + n_q]
            out.append(q.transpose(0, 2, 1))
        return np.concatenate(out, axis=0).astype(np.float32)
