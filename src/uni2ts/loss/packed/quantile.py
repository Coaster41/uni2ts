import abc
from typing import Callable, Optional

import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int

from uni2ts.common.core import abstract_class_property

from ._base import PackedQuantileLoss


@abstract_class_property("error_func")
class PackedQuantileLoss(PackedQuantileLoss, abc.ABC):
    error_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = NotImplemented

    def __init__(
        self,
        quantile_levels: tuple[Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ):
        super().__init__()
        self.quantile_levels = quantile_levels

    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len num_quantiles*patch_size"],
        target: Float[torch.Tensor, "*batch seq_len patch_size"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch_size"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len patch_size"]:
        quantile_levels = torch.tensor(self.quantile_levels, device=pred.device).view(
            1, 1, -1, 1
        )
        pred = rearrange(
            pred,
            "... (num_quantiles patch_size) -> ... num_quantiles patch_size",
            num_quantiles=len(self.quantile_levels),
        )
        target = repeat(
            target,
            "... patch_size -> ... num_quantiles patch_size",
            num_quantiles=len(self.quantile_levels),
        )
        errors = self.error_func(pred, target)
        indicator = target > pred

        quantile_loss = torch.where(
            indicator, quantile_levels * errors, (1 - quantile_levels) * errors
        )
        # aggregated by num_quantile axis
        return quantile_loss.mean(dim=-2)


class PackedQuantileMAELoss(PackedQuantileLoss):
    error_func = torch.nn.L1Loss(reduction="none")


class PackedQuantileMTPLoss(PackedQuantileLoss):
    """Multi-token-prediction quantile MAE loss.

    A single implementation that covers any ``(num_predict_token, objective)``
    combination. ``num_predict_token`` (npt) is inferred from the prediction
    tensor, so npt == 1 and npt > 1 both work. The objective is selected by
    ``shift``:

      - ``shift=False`` (current-token / encoder / masked reconstruction):
        position ``t``'s prediction targets patches ``t .. t+npt-1``.
      - ``shift=True`` (next-token / decoder / autoregressive):
        position ``t``'s prediction targets patches ``t+1 .. t+npt``.

    ``shift=True`` reproduces the former ``PackedQuantileDecoderMAELoss`` and
    ``shift=False`` (with npt == 1) reproduces ``PackedQuantileEncoderMAELoss``.
    """

    error_func = torch.nn.L1Loss(reduction="none")
    clamp_loss = 3

    def __init__(
        self,
        quantile_levels: tuple[Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        shift: bool = False,
    ):
        super().__init__(quantile_levels)
        self.shift = shift

    def _loss_func(
        self,
        pred: Float[
            torch.Tensor, "*batch seq_len num_predict_token*num_quantiles*patch_size"
        ],
        target: Float[torch.Tensor, "*batch seq_len patch_size"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch_size"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch aligned_seq_len num_predict_token*patch_size"]:
        num_quantiles = len(self.quantile_levels)
        patch_size = target.shape[-1]
        preds = rearrange(
            pred,
            "... seq_len (predict_token num_quantiles patch_size) -> ... seq_len num_quantiles (predict_token patch_size)",
            num_quantiles=num_quantiles,
            patch_size=patch_size,
        )
        num_predict_token = preds.shape[-1] // patch_size
        # *batch seq_len num_quantiles (num_pred_tok patch_size)

        # Rolling windows of npt consecutive target patches.
        targets = target.unfold(-2, num_predict_token, 1)
        # *batch (seq_len-num_pred_tok+1) patch_size num_pred_tok
        if self.shift:
            # next-token: align position t's pred with the window starting at t+1.
            preds = preds[..., :-num_predict_token, :, :]
            targets = targets[..., 1:, :, :]
        else:
            # current-token: align position t's pred with the window starting at t.
            preds = preds[..., : targets.shape[-3], :, :]
        targets = repeat(
            targets,
            "... patch_size predict_token -> ... num_quantiles (predict_token patch_size)",
            num_quantiles=num_quantiles,
        )

        quantile_levels = torch.tensor(self.quantile_levels, device=pred.device).view(
            1, 1, -1, 1
        )
        errors = self.error_func(preds, targets)
        indicator = targets > preds

        quantile_loss = torch.where(
            indicator, quantile_levels * errors, (1 - quantile_levels) * errors
        )
        # aggregate over the quantile axis
        quantile_loss = quantile_loss.clamp(max=self.clamp_loss)
        return quantile_loss.mean(dim=-2)

    def reduce_loss(
        self,
        loss: Float[
            torch.Tensor, "*batch aligned_seq_len num_predict_token*patch_size"
        ],
        prediction_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]],
        observed_mask: Optional[Bool[torch.Tensor, "*batch seq_len patch_size"]],
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]],
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]],
    ) -> Float[torch.Tensor, ""]:
        patch_size = observed_mask.shape[-1]
        num_predict_token = loss.shape[-1] // patch_size
        mask = prediction_mask.unsqueeze(-1) * observed_mask
        # ASSUMING UNIVARIATE WITH NO COVARIATES
        mask = mask.unfold(-2, num_predict_token, 1)
        if self.shift:
            mask = mask[..., 1:, :, :]
        mask = rearrange(
            mask, "... patch_size num_pred_token -> ... (num_pred_token patch_size)"
        )
        loss = loss.clamp(max=self.clamp_loss)
        masked_loss = (loss * mask).sum()
        num_active = mask.sum().clamp(min=1)  # avoid division by zero
        return masked_loss / num_active


class PackedQuantileDecoderMAELoss(PackedQuantileMTPLoss):
    """Back-compat alias: next-token (AR) objective. Use ``PackedQuantileMTPLoss(shift=True)``."""

    def __init__(
        self,
        quantile_levels: tuple[Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ):
        super().__init__(quantile_levels, shift=True)


class PackedQuantileEncoderMAELoss(PackedQuantileMTPLoss):
    """Back-compat alias: current-token (masked reconstruction) objective. ``shift=False``."""

    def __init__(
        self,
        quantile_levels: tuple[Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ):
        super().__init__(quantile_levels, shift=False)
