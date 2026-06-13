#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import math
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Generator, List, Optional

import lightning as L
import numpy
import numpy as np
import torch
from einops import rearrange, reduce, repeat
from gluonts.model import Input, InputSpec
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.torch import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    CausalMeanValueImputation,
    ExpandDimArray,
    TestSplitSampler,
    Transformation,
)
from gluonts.transform.split import TFTInstanceSplitter
from jaxtyping import Bool, Float, Int

from uni2ts.transform.imputation import CausalMeanImputation

from .module import MoiraiXModule


class MoiraiXForecast(L.LightningModule):
    """Unified forecaster for :class:`MoiraiXModule`.

    Dispatches to one of two autoregressive strategies based on the model's
    objective (``module.predict_next``):

      - next-token (causal decoder): cache-aware AR, predictions read from the
        token preceding each slot; supports ``naive`` / ``branch`` / ``trajectory``.
      - current-token (reconstruction encoder): full-resweep AR, predictions read
        at the masked slot; supports ``naive`` / ``branch``.

    ``single_pass_horizon`` controls how far ahead (in time steps) the first single
    forward pass commits before AR begins (see ``__init__``).
    """

    #: Module class instantiated from ``module_kwargs``. Preset shims override this.
    module_class: type = MoiraiXModule

    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        feat_dynamic_real_dim: int,
        past_feat_dynamic_real_dim: int,
        context_length: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[MoiraiXModule] = None,
        use_cache: bool = True,
        update_context_in_ar: Optional[bool] = None,
        ar_method: Optional[str] = None,
        ar_num_trajectories: int = 128,
        ar_u_scheme: Optional[str] = "iid",
        ar_num_patches: int = 1,
        single_pass_horizon: Optional[int] = None,
        distribution: Optional[Any] = None,
    ):
        """
        :param single_pass_horizon: how far ahead (in time steps) the model forecasts
            in the first single forward pass before switching to AR. ``-1`` means
            single pass only (no AR; predict the whole horizon at once, natively
            supported for current-token models). ``None`` resolves to
            ``num_predict_token`` patches if ``predict_next`` (reproduces the decoder)
            else ``-1`` (reproduces the encoder).
        :param use_cache: KV-cache the context during AR (next-token / causal only).
        :param ar_method: ``None`` | ``"naive"`` | ``"branch"`` | ``"trajectory"``.
        :param ar_num_patches: patches committed per AR step (current-token path).
        :param ar_num_trajectories / ar_u_scheme / distribution: trajectory-AR config.
        """
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        if module_kwargs and "attn_dropout_p" in module_kwargs:
            module_kwargs["attn_dropout_p"] = 0
        if module_kwargs and "dropout_p" in module_kwargs:
            module_kwargs["dropout_p"] = 0

        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "module",
                "use_cache",
                "update_context_in_ar",
                "ar_method",
                "ar_num_trajectories",
                "ar_u_scheme",
                "ar_num_patches",
                "single_pass_horizon",
                "distribution",
            ]
        )
        self.module = self.module_class(**module_kwargs) if module is None else module
        self.module.eval()

        assert ar_method in (None, "naive", "branch", "trajectory")
        self.ar_method = ar_method
        self.ar_num_trajectories = ar_num_trajectories
        self.ar_u_scheme = ar_u_scheme
        self.ar_num_patches = ar_num_patches
        assert ar_num_patches >= 1
        self.single_pass_horizon = single_pass_horizon

        # KV cache is only meaningful for a causal model.
        if use_cache and not self.module.causal:
            use_cache = False
        self.use_cache = use_cache

        if update_context_in_ar is None:
            update_context_in_ar = not use_cache
        self.update_context_in_ar = update_context_in_ar
        if self.use_cache and self.update_context_in_ar:
            raise ValueError(
                "use_cache=True is incompatible with update_context_in_ar=True: "
                "the cache freezes loc/scale at context-only and cannot represent "
                "AR commits being folded into the scaler. Set one of them to False."
            )

        if ar_method == "trajectory" and not self.module.predict_next:
            raise ValueError(
                "ar_method='trajectory' is only supported for next-token "
                "(predict_next=True) models."
            )
        if distribution is None and ar_method == "trajectory":
            from uni2ts.module.distributions import QuantileKnotDistribution

            distribution = lambda: QuantileKnotDistribution(tails="gaussian")
        self.distribution = distribution

    # ------------------------------------------------------------------ #
    # single-pass horizon resolution                                     #
    # ------------------------------------------------------------------ #
    def _single_pass_tokens(self, per_var_predict_token: int, patch_size: int) -> int:
        """Number of prediction *tokens* committed from the first forward pass."""
        sph = self.single_pass_horizon
        if sph is None:
            return (
                self.module.num_predict_token
                if self.module.predict_next
                else per_var_predict_token
            )
        if sph == -1:
            return per_var_predict_token
        return max(1, math.ceil(sph / patch_size))

    @contextmanager
    def hparams_context(
        self,
        prediction_length: Optional[int] = None,
        target_dim: Optional[int] = None,
        feat_dynamic_real_dim: Optional[int] = None,
        past_feat_dynamic_real_dim: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> Generator["MoiraiXForecast", None, None]:
        kwargs = {
            "prediction_length": prediction_length,
            "target_dim": target_dim,
            "feat_dynamic_real_dim": feat_dynamic_real_dim,
            "past_feat_dynamic_real_dim": past_feat_dynamic_real_dim,
            "context_length": context_length,
        }
        old_hparams = deepcopy(self.hparams)
        for kw, arg in kwargs.items():
            if arg is not None:
                self.hparams[kw] = arg

        yield self

        for kw in kwargs:
            self.hparams[kw] = old_hparams[kw]

    def create_predictor(
        self,
        batch_size: int,
        device: str = "auto",
    ) -> PyTorchPredictor:
        ts_fields = []
        if self.hparams.feat_dynamic_real_dim > 0:
            ts_fields.append("feat_dynamic_real")
            ts_fields.append("observed_feat_dynamic_real")
        past_ts_fields = []
        if self.hparams.past_feat_dynamic_real_dim > 0:
            past_ts_fields.append("past_feat_dynamic_real")
            past_ts_fields.append("past_observed_feat_dynamic_real")
        instance_splitter = TFTInstanceSplitter(
            instance_sampler=TestSplitSampler(),
            past_length=self.past_length,
            future_length=self.hparams.prediction_length,
            observed_value_field="observed_target",
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )
        return PyTorchPredictor(
            input_names=self.prediction_input_names,
            prediction_net=self,
            batch_size=batch_size,
            prediction_length=self.hparams.prediction_length,
            input_transform=self.get_default_transform() + instance_splitter,
            forecast_generator=QuantileForecastGenerator(self.module.quantile_levels),
            device=device,
        )

    def describe_inputs(self, batch_size: int = 1) -> InputSpec:
        data = {
            "past_target": Input(
                shape=(batch_size, self.past_length, self.hparams.target_dim),
                dtype=torch.float,
            ),
            "past_observed_target": Input(
                shape=(batch_size, self.past_length, self.hparams.target_dim),
                dtype=torch.bool,
            ),
            "past_is_pad": Input(
                shape=(batch_size, self.past_length),
                dtype=torch.bool,
            ),
        }
        if self.hparams.feat_dynamic_real_dim > 0:
            data["feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length + self.hparams.prediction_length,
                    self.hparams.feat_dynamic_real_dim,
                ),
                dtype=torch.float,
            )
            data["observed_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length + self.hparams.prediction_length,
                    self.hparams.feat_dynamic_real_dim,
                ),
                dtype=torch.bool,
            )
        if self.hparams.past_feat_dynamic_real_dim > 0:
            data["past_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.past_feat_dynamic_real_dim,
                ),
                dtype=torch.float,
            )
            data["past_observed_feat_dynamic_real"] = Input(
                shape=(
                    batch_size,
                    self.past_length,
                    self.hparams.past_feat_dynamic_real_dim,
                ),
                dtype=torch.bool,
            )
        return InputSpec(data=data, zeros_fn=torch.zeros)

    @property
    def prediction_input_names(self) -> list[str]:
        return list(self.describe_inputs())

    @property
    def training_input_names(self):
        return self.prediction_input_names + ["future_target", "future_observed_values"]

    @property
    def past_length(self) -> int:
        return self.hparams.context_length

    def context_token_length(self, patch_size: int) -> int:
        return math.ceil(self.hparams.context_length / patch_size)

    def prediction_token_length(self, patch_size) -> int:
        return math.ceil(self.hparams.prediction_length / patch_size)

    @property
    def max_patch_size(self) -> int:
        return max(self.module.patch_sizes)

    # ------------------------------------------------------------------ #
    # forward dispatch                                                   #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> Float[torch.Tensor, "batch num_quantiles future_time *tgt"]:
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            self.module.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )
        args = (target, observed_mask, sample_id, time_id, variate_id, prediction_mask)
        if self.module.predict_next:
            return self._forecast_next_token(*args)
        return self._forecast_current_token(*args)

    # ------------------------------------------------------------------ #
    # next-token (causal decoder) AR — ported from MoiraicForecast       #
    # ------------------------------------------------------------------ #
    def _forecast_next_token(
        self, target, observed_mask, sample_id, time_id, variate_id, prediction_mask
    ):
        qlevels = list(self.module.quantile_levels)
        median_idx = min(range(len(qlevels)), key=lambda i: abs(qlevels[i] - 0.5))

        per_var_context_token = self.context_token_length(self.module.patch_size)
        total_context_token = self.hparams.target_dim * per_var_context_token
        per_var_predict_token = self.prediction_token_length(self.module.patch_size)
        total_predict_token = self.hparams.target_dim * per_var_predict_token

        pred_index = torch.arange(
            start=per_var_context_token - 1,
            end=total_context_token,
            step=per_var_context_token,
        )
        assign_index = torch.arange(
            start=total_context_token,
            end=total_context_token + total_predict_token,
            step=per_var_predict_token,
        )
        quantile_prediction = repeat(
            target,
            "... patch_size -> ... num_quantiles patch_size",
            num_quantiles=len(self.module.quantile_levels),
            patch_size=self.module.patch_size,
        ).clone()

        # A pure causal decoder can emit at most num_predict_token patches per pass.
        spt = self._single_pass_tokens(per_var_predict_token, self.module.patch_size)
        if spt > self.module.num_predict_token:
            warnings.warn(
                f"single_pass_horizon requests {spt} tokens but a next-token model "
                f"emits at most num_predict_token={self.module.num_predict_token} per "
                f"pass; clamping."
            )
            spt = self.module.num_predict_token
        will_ar = per_var_predict_token > spt
        if will_ar and self.ar_method is None:
            raise ValueError(
                "A next-token model needs an ar_method ('naive'/'branch'/'trajectory') "
                f"to forecast {per_var_predict_token} prediction tokens beyond its "
                f"single-pass capacity of {spt}. (single_pass_horizon={self.single_pass_horizon})"
            )
        use_cache = self.use_cache and will_ar

        # ---------------- Prefill ----------------
        if use_cache:
            preds, cache = self.module(
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
                training_mode=False,
                return_cache=True,
            )
        else:
            preds = self.module(
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
                training_mode=False,
            )

        def structure_multi_predict(
            per_var_predict_token, pred_index, assign_index, preds
        ):
            preds = rearrange(
                preds,
                "... (predict_token num_quantiles patch_size) -> ... predict_token num_quantiles patch_size",
                predict_token=self.module.num_predict_token,
                num_quantiles=self.module.num_quantiles,
                patch_size=self.module.patch_size,
            )
            preds = rearrange(
                preds[..., pred_index, :per_var_predict_token, :, :],
                "... pred_index predict_token num_quantiles patch_size -> ... (pred_index predict_token) num_quantiles patch_size",
            )
            adjusted_assign_index = torch.cat(
                [
                    torch.arange(start=idx, end=idx + per_var_predict_token)
                    for idx in assign_index
                ]
            )
            return preds, adjusted_assign_index

        # ---------------- Single-shot exit (no AR) ----------------
        if not will_ar:
            preds, adjusted_assign_index = structure_multi_predict(
                per_var_predict_token,
                pred_index,
                assign_index,
                preds,
            )
            quantile_prediction[..., adjusted_assign_index, :, :] = preds
            return self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )

        if self.ar_method == "naive":
            preds, adjusted_assign_index = structure_multi_predict(
                self.module.num_predict_token,
                pred_index,
                assign_index,
                preds,
            )
            quantile_prediction[..., adjusted_assign_index, :, :] = preds
            target[..., adjusted_assign_index, :] = preds[..., median_idx, :]

            if self.update_context_in_ar:
                prediction_mask[..., adjusted_assign_index] = False

            remain_step = per_var_predict_token - self.module.num_predict_token
            while remain_step > 0:
                if use_cache:
                    preds = self.module(
                        target[..., total_context_token:, :],
                        observed_mask[..., total_context_token:, :],
                        sample_id[..., total_context_token:],
                        time_id[..., total_context_token:],
                        variate_id[..., total_context_token:],
                        prediction_mask[..., total_context_token:],
                        training_mode=False,
                        past_cache=cache,
                    )
                else:
                    preds = self.module(
                        target,
                        observed_mask,
                        sample_id,
                        time_id,
                        variate_id,
                        prediction_mask,
                        training_mode=False,
                    )

                pred_index = assign_index + self.module.num_predict_token - 1
                assign_index = pred_index + 1
                pred_index_for_struct = (
                    pred_index - total_context_token if use_cache else pred_index
                )

                step_size = (
                    self.module.num_predict_token
                    if remain_step - self.module.num_predict_token > 0
                    else remain_step
                )
                preds, adjusted_assign_index = structure_multi_predict(
                    step_size,
                    pred_index_for_struct,
                    assign_index,
                    preds,
                )
                quantile_prediction[..., adjusted_assign_index, :, :] = preds
                target[..., adjusted_assign_index, :] = preds[..., median_idx, :]

                if self.update_context_in_ar:
                    prediction_mask[..., adjusted_assign_index] = False

                remain_step -= self.module.num_predict_token

            return self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )

        if self.ar_method == "trajectory":
            N = self.ar_num_trajectories
            qlevels_t = torch.tensor(
                self.module.quantile_levels,
                device=self.device,
                dtype=torch.float32,
            )

            def exp(t):
                return repeat(t, "b ... -> b n ...", n=N).clone()

            e_target = exp(target)
            e_observed_mask = exp(observed_mask)
            e_sample_id = exp(sample_id)
            e_time_id = exp(time_id)
            e_variate_id = exp(variate_id)
            e_prediction_mask = exp(prediction_mask)

            if use_cache:
                cache = self._expand_cache(cache, factor=N)

            preds_struct, adjusted_assign_index = structure_multi_predict(
                self.module.num_predict_token,
                pred_index,
                assign_index,
                preds,
            )
            preds_bn = repeat(preds_struct, "b pt q p -> b n pt q p", n=N)
            samples = self._sample_from_quantile_forecast(preds_bn)

            e_target[..., adjusted_assign_index, :] = samples
            if self.update_context_in_ar:
                e_prediction_mask[..., adjusted_assign_index] = False

            remain_step = per_var_predict_token - self.module.num_predict_token
            while remain_step > 0:
                if use_cache:
                    preds = self.module(
                        e_target[..., total_context_token:, :],
                        e_observed_mask[..., total_context_token:, :],
                        e_sample_id[..., total_context_token:],
                        e_time_id[..., total_context_token:],
                        e_variate_id[..., total_context_token:],
                        e_prediction_mask[..., total_context_token:],
                        training_mode=False,
                        past_cache=cache,
                    )
                else:
                    preds = self.module(
                        e_target,
                        e_observed_mask,
                        e_sample_id,
                        e_time_id,
                        e_variate_id,
                        e_prediction_mask,
                        training_mode=False,
                    )

                pred_index = assign_index + self.module.num_predict_token - 1
                assign_index = pred_index + 1
                pred_index_for_struct = (
                    pred_index - total_context_token if use_cache else pred_index
                )
                step_size = (
                    self.module.num_predict_token
                    if remain_step - self.module.num_predict_token > 0
                    else remain_step
                )
                preds_struct, adjusted_assign_index = structure_multi_predict(
                    step_size,
                    pred_index_for_struct,
                    assign_index,
                    preds,
                )
                samples = self._sample_from_quantile_forecast(preds_struct)

                e_target[..., adjusted_assign_index, :] = samples
                if self.update_context_in_ar:
                    e_prediction_mask[..., adjusted_assign_index] = False
                remain_step -= self.module.num_predict_token

            pred_slice = slice(
                total_context_token, total_context_token + total_predict_token
            )
            samples_all = e_target[..., pred_slice, :]
            q_out = torch.quantile(samples_all, qlevels_t, dim=1)
            q_out = rearrange(q_out, "q b seq p -> b seq q p")
            quantile_prediction[..., pred_slice, :, :] = q_out

            return self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )

        if self.ar_method == "branch":

            def exp_q(t):
                return repeat(
                    t,
                    "batch_size ...  -> batch_size num_quantiles ...",
                    num_quantiles=len(self.module.quantile_levels),
                    batch_size=t.shape[0],
                ).clone()

            expand_target = exp_q(target)
            expand_prediction_mask = exp_q(prediction_mask)
            expand_observed_mask = exp_q(observed_mask)
            expand_sample_id = exp_q(sample_id)
            expand_time_id = exp_q(time_id)
            expand_variate_id = exp_q(variate_id)

            if use_cache:
                cache = self._expand_cache(
                    cache, factor=len(self.module.quantile_levels)
                )

            preds, adjusted_assign_index = structure_multi_predict(
                self.module.num_predict_token,
                pred_index,
                assign_index,
                preds,
            )
            quantile_prediction[..., adjusted_assign_index, :, :] = preds
            expand_target[..., adjusted_assign_index, :] = rearrange(
                preds,
                "... predict_token num_quantiles patch_size -> ... num_quantiles predict_token patch_size",
                num_quantiles=self.module.num_quantiles,
                patch_size=self.module.patch_size,
                predict_token=self.module.num_predict_token,
            )

            if self.update_context_in_ar:
                expand_prediction_mask[..., adjusted_assign_index] = False

            remain_step = per_var_predict_token - self.module.num_predict_token
            while remain_step > 0:
                if use_cache:
                    preds = self.module(
                        expand_target[..., total_context_token:, :],
                        expand_observed_mask[..., total_context_token:, :],
                        expand_sample_id[..., total_context_token:],
                        expand_time_id[..., total_context_token:],
                        expand_variate_id[..., total_context_token:],
                        expand_prediction_mask[..., total_context_token:],
                        training_mode=False,
                        past_cache=cache,
                    )
                else:
                    preds = self.module(
                        expand_target,
                        expand_observed_mask,
                        expand_sample_id,
                        expand_time_id,
                        expand_variate_id,
                        expand_prediction_mask,
                        training_mode=False,
                    )

                pred_index = assign_index + self.module.num_predict_token - 1
                assign_index = pred_index + 1
                pred_index_for_struct = (
                    pred_index - total_context_token if use_cache else pred_index
                )

                preds, adjusted_assign_index = structure_multi_predict(
                    (
                        self.module.num_predict_token
                        if remain_step - self.module.num_predict_token > 0
                        else remain_step
                    ),
                    pred_index_for_struct,
                    assign_index,
                    preds,
                )
                quantile_prediction_next_step = rearrange(
                    preds,
                    "... num_quantiles_prev pred_index num_quantiles patch_size -> ... pred_index (num_quantiles_prev num_quantiles) patch_size",
                    num_quantiles=self.module.num_quantiles,
                    patch_size=self.module.patch_size,
                )
                quantile_prediction_next_step = torch.quantile(
                    quantile_prediction_next_step,
                    torch.tensor(
                        self.module.quantile_levels,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    dim=-2,
                )
                quantile_prediction[..., adjusted_assign_index, :, :] = rearrange(
                    quantile_prediction_next_step,
                    "num_quantiles ... patch_size -> ... num_quantiles patch_size",
                )

                expand_target[..., adjusted_assign_index, :] = rearrange(
                    quantile_prediction_next_step,
                    "num_quantiles batch_size predict_token patch_size -> batch_size num_quantiles predict_token patch_size",
                    num_quantiles=self.module.num_quantiles,
                    patch_size=self.module.patch_size,
                    predict_token=len(adjusted_assign_index),
                )
                if self.update_context_in_ar:
                    expand_prediction_mask[..., adjusted_assign_index] = False

                remain_step -= self.module.num_predict_token

            return self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )

    # ------------------------------------------------------------------ #
    # current-token (reconstruction encoder) AR — from MoiraieForecast   #
    # ------------------------------------------------------------------ #
    def _forecast_current_token(
        self, target, observed_mask, sample_id, time_id, variate_id, prediction_mask
    ):
        per_var_context_token = self.context_token_length(self.module.patch_size)
        total_context_token = self.hparams.target_dim * per_var_context_token
        per_var_predict_token = self.prediction_token_length(self.module.patch_size)
        total_predict_token = self.hparams.target_dim * per_var_predict_token
        Q = len(self.module.quantile_levels)

        pred_index = torch.arange(
            start=total_context_token,
            end=total_context_token + total_predict_token,
            step=1,
        )
        quantile_prediction = repeat(
            target,
            "... patch_size -> ... num_quantiles patch_size",
            num_quantiles=Q,
            patch_size=self.module.patch_size,
        ).clone()

        def commit_indices(committed, step):
            return torch.cat(
                [
                    torch.arange(
                        total_context_token + v * per_var_predict_token + committed,
                        total_context_token
                        + v * per_var_predict_token
                        + committed
                        + step,
                        device=target.device,
                    )
                    for v in range(self.hparams.target_dim)
                ]
            )

        def run_and_extract(t, om, sid, tid, vid, pm):
            preds = self.module(t, om, sid, tid, vid, pm, training_mode=False)
            return rearrange(
                preds[..., pred_index, :],
                "... seq (predict_token num_quantiles patch_size) "
                "-> ... (seq predict_token) num_quantiles patch_size",
                predict_token=self.module.num_predict_token,
                num_quantiles=self.module.num_quantiles,
                patch_size=self.module.patch_size,
            )

        # How many tokens to commit from the first single forward pass.
        spt = self._single_pass_tokens(per_var_predict_token, self.module.patch_size)
        spt = min(spt, per_var_predict_token)
        will_ar = (self.ar_method is not None) and (spt < per_var_predict_token)

        # ---------------- Single-shot (default) ----------------
        if not will_ar:
            preds_at_pred = run_and_extract(
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
            )
            # run_and_extract interleaves (seq * predict_token); for the
            # current-token encoder each position's pt_idx=0 is the direct
            # prediction for that position, so stride-slice away the extras.
            npt = self.module.num_predict_token
            if npt > 1:
                preds_at_pred = preds_at_pred[..., ::npt, :, :]
            quantile_prediction[..., pred_index, :, :] = preds_at_pred
            return self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )

        # ---------------- Naive AR ----------------
        if self.ar_method == "naive":
            qlevels = list(self.module.quantile_levels)
            median_idx = min(range(Q), key=lambda i: abs(qlevels[i] - 0.5))

            committed = 0
            first = True
            while committed < per_var_predict_token:
                preds_at_pred = run_and_extract(
                    target,
                    observed_mask,
                    sample_id,
                    time_id,
                    variate_id,
                    prediction_mask,
                )
                this_step = spt if first else self.ar_num_patches
                step = min(this_step, per_var_predict_token - committed)
                first = False
                commit = commit_indices(committed, step)
                slot = commit - total_context_token

                preds_commit = preds_at_pred[..., slot, :, :]
                quantile_prediction[..., commit, :, :] = preds_commit
                target[..., commit, :] = preds_commit[..., median_idx, :]
                prediction_mask[..., commit] = False
                observed_mask[..., commit, :] = True
                committed += step

            return self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )

        # ---------------- Branching AR ----------------
        if self.ar_method == "branch":
            preds_at_pred = run_and_extract(
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
            )
            step = min(spt, per_var_predict_token)
            commit = commit_indices(0, step)
            slot = commit - total_context_token
            preds_commit = preds_at_pred[..., slot, :, :]
            quantile_prediction[..., commit, :, :] = preds_commit

            def expand(t):
                return repeat(t, "b ... -> b q ...", q=Q).clone()

            e_target = expand(target)
            e_observed_mask = expand(observed_mask)
            e_sample_id = expand(sample_id)
            e_time_id = expand(time_id)
            e_variate_id = expand(variate_id)
            e_prediction_mask = expand(prediction_mask)

            e_target[..., commit, :] = rearrange(
                preds_commit, "b commit q p -> b q commit p"
            )
            e_prediction_mask[..., commit] = False
            e_observed_mask[..., commit, :] = True

            committed = step
            while committed < per_var_predict_token:
                step = min(self.ar_num_patches, per_var_predict_token - committed)
                commit = commit_indices(committed, step)
                slot = commit - total_context_token

                preds_at_pred = run_and_extract(
                    e_target,
                    e_observed_mask,
                    e_sample_id,
                    e_time_id,
                    e_variate_id,
                    e_prediction_mask,
                )
                preds_commit = preds_at_pred[..., slot, :, :]

                merged = rearrange(preds_commit, "b qp commit q p -> b commit (qp q) p")
                agg = torch.quantile(
                    merged,
                    torch.tensor(
                        self.module.quantile_levels,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    dim=-2,
                )
                agg = rearrange(agg, "q b commit p -> b commit q p")

                quantile_prediction[..., commit, :, :] = agg
                e_target[..., commit, :] = rearrange(
                    agg, "b commit q p -> b q commit p"
                )
                e_prediction_mask[..., commit] = False
                e_observed_mask[..., commit, :] = True
                committed += step

            return self._format_preds(
                self.module.num_quantiles,
                self.module.patch_size,
                quantile_prediction,
                self.hparams.target_dim,
            )

    def predict(
        self,
        past_target: List[Float[np.ndarray, "past_time tgt"]],
        feat_dynamic_real: Optional[
            List[Float[np.ndarray, "batch past_time tgt"]]
        ] = None,
        past_feat_dynamic_real: Optional[
            List[Float[np.ndarray, "batch past_time tgt"]]
        ] = None,
    ) -> Float[numpy.ndarray, "batch num_quantiles future_time *tgt"]:
        data_entry = {
            "past_target": past_target,
            "feat_dynamic_real": feat_dynamic_real,
            "past_feat_dynamic_real": past_feat_dynamic_real,
        }

        data_entry["past_observed_target"] = [~np.isnan(x) for x in past_target]
        if feat_dynamic_real:
            data_entry["observed_feat_dynamic_real"] = [
                ~np.isnan(x) for x in feat_dynamic_real
            ]
        else:
            data_entry["observed_feat_dynamic_real"] = None

        if past_feat_dynamic_real:
            data_entry["past_observed_feat_dynamic_real"] = [
                ~np.isnan(x) for x in past_feat_dynamic_real
            ]
        else:
            data_entry["past_observed_feat_dynamic_real"] = None

        impute = CausalMeanImputation()

        def process_sample(sample):
            arr = np.asarray(sample)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            if np.issubdtype(arr.dtype, np.number) and np.isnan(arr).any():
                arr = impute(arr)
            return arr

        for key, value in data_entry.items():
            if value is not None:
                data_entry[key] = [process_sample(sample) for sample in value]

        batch_size = len(data_entry["past_target"])

        data_entry["past_is_pad"] = np.zeros(
            (batch_size, self.hparams.context_length), dtype=bool
        )

        context_length = self.hparams.context_length
        prediction_length = self.hparams.prediction_length
        full_length = context_length + prediction_length

        context_only_keys = {
            "past_target",
            "past_observed_target",
            "past_feat_dynamic_real",
            "past_observed_feat_dynamic_real",
        }
        full_horizon_keys = {"feat_dynamic_real", "observed_feat_dynamic_real"}

        for key in context_only_keys:
            if data_entry.get(key) is not None:
                for idx in range(len(data_entry[key])):
                    if data_entry[key][idx].shape[0] > context_length:
                        data_entry[key][idx] = data_entry[key][idx][-context_length:, :]
                    else:
                        pad_length = context_length - data_entry[key][idx].shape[0]
                        pad_block = np.full(
                            (pad_length, data_entry[key][idx].shape[-1]),
                            data_entry[key][idx][0],
                            dtype=data_entry[key][idx].dtype,
                        )
                        data_entry[key][idx] = np.concatenate(
                            [pad_block, data_entry[key][idx]], axis=0
                        )
                        if key == "past_target":
                            data_entry["past_is_pad"][idx, :pad_length] = True

        for key in full_horizon_keys:
            if data_entry.get(key) is not None:
                for idx in range(len(data_entry[key])):
                    if data_entry[key][idx].shape[0] > full_length:
                        data_entry[key][idx] = data_entry[key][idx][-full_length:, :]
                    elif data_entry[key][idx].shape[0] < full_length:
                        pad_length = full_length - data_entry[key][idx].shape[0]
                        pad_block = np.full(
                            (pad_length, data_entry[key][idx].shape[-1]),
                            data_entry[key][idx][0],
                            dtype=data_entry[key][idx].dtype,
                        )
                        data_entry[key][idx] = np.concatenate(
                            [pad_block, data_entry[key][idx]], axis=0
                        )

        for k in ["past_target", "feat_dynamic_real", "past_feat_dynamic_real"]:
            if data_entry.get(k) is not None:
                data_entry[k] = torch.tensor(
                    np.array(data_entry[k]), device=self.device, dtype=torch.float32
                )

        for k in [
            "past_observed_target",
            "observed_feat_dynamic_real",
            "past_observed_feat_dynamic_real",
            "past_is_pad",
        ]:
            if data_entry.get(k) is not None:
                data_entry[k] = torch.tensor(
                    np.array(data_entry[k]), device=self.device, dtype=torch.bool
                )

        with torch.no_grad():
            predictions = (
                self(
                    data_entry["past_target"],
                    data_entry["past_observed_target"],
                    data_entry["past_is_pad"],
                    feat_dynamic_real=data_entry.get("feat_dynamic_real"),
                    observed_feat_dynamic_real=data_entry.get(
                        "observed_feat_dynamic_real"
                    ),
                    past_feat_dynamic_real=data_entry.get("past_feat_dynamic_real"),
                    past_observed_feat_dynamic_real=data_entry.get(
                        "past_observed_feat_dynamic_real"
                    ),
                )
                .detach()
                .cpu()
                .numpy()
            )
        return predictions

    def _draw_u(self, shape, device, dtype, scheme="iid", generator=None):
        B, M, T, P = shape
        if scheme == "iid":
            return torch.rand(shape, device=device, dtype=dtype, generator=generator)
        if scheme == "stratified":
            base = torch.arange(M, device=device, dtype=dtype).view(1, M, 1, 1)
            jitter = torch.rand(shape, device=device, dtype=dtype, generator=generator)
            return (base + jitter) / M
        if scheme == "antithetic":
            assert M % 2 == 0
            half = torch.rand(
                (B, M // 2, T, P), device=device, dtype=dtype, generator=generator
            )
            return torch.cat([half, 1.0 - half], dim=1)
        if scheme == "quantile":
            q = torch.tensor(self.module.quantile_levels, device=device, dtype=dtype)
            assert M == q.numel()
            return q.view(1, M, 1, 1).expand(shape).contiguous()
        if scheme == "median":
            return torch.full(shape, 0.5, device=device, dtype=dtype)
        raise ValueError(scheme)

    def _sample_from_quantile_forecast(self, preds: torch.Tensor) -> torch.Tensor:
        values = rearrange(preds, "... predict_token q p -> ... predict_token p q")
        dist = self.distribution()
        dist.fit(self.module.quantile_levels, values.detach())
        u = self._draw_u(
            values.shape[:-1],
            device=values.device,
            dtype=values.dtype,
            scheme=self.ar_u_scheme,
        )
        return dist.ppf(u).to(dtype=preds.dtype)

    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: Optional[float] = None,
    ) -> torch.Tensor:
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)

    def _generate_time_id(
        self,
        patch_size: int,
        past_observed_target: Bool[torch.Tensor, "batch past_seq tgt"],
    ) -> tuple[
        Int[torch.Tensor, "batch past_token"], Int[torch.Tensor, "batch future_token"]
    ]:
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(
            past_seq_id.cummax(dim=-1).values.cumsum(dim=-1) - 1, min=0
        )
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        future_seq_id = (
            repeat(
                torch.arange(
                    self.prediction_token_length(patch_size),
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )
        return past_seq_id, future_seq_id

    def _convert(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        future_target: Optional[Float[torch.Tensor, "batch future_time tgt"]] = None,
        future_observed_target: Optional[
            Bool[torch.Tensor, "batch future_time tgt"]
        ] = None,
        future_is_pad: Optional[Bool[torch.Tensor, "batch future_time"]] = None,
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> tuple[
        Float[torch.Tensor, "batch combine_seq patch"],
        Bool[torch.Tensor, "batch combine_seq patch"],
        Int[torch.Tensor, "batch combine_seq"],
        Int[torch.Tensor, "batch combine_seq"],
        Int[torch.Tensor, "batch combine_seq"],
        Bool[torch.Tensor, "batch combine_seq"],
    ]:
        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []
        prediction_mask = []
        dim_count = 0

        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target
        )

        if future_target is None:
            future_target = torch.zeros(
                batch_shape + (self.hparams.prediction_length, past_target.shape[-1]),
                dtype=past_target.dtype,
                device=device,
            )
        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(patch_size, past_target, -2, left=True),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
            ]
        )
        if future_observed_target is None:
            future_observed_target = torch.ones(
                batch_shape
                + (self.hparams.prediction_length, past_observed_target.shape[-1]),
                dtype=torch.bool,
                device=device,
            )
        observed_mask.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_target, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_observed_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
            ]
        )
        if future_is_pad is None:
            future_is_pad = torch.zeros(
                batch_shape + (self.hparams.prediction_length,),
                dtype=torch.long,
                device=device,
            )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=self.prediction_token_length(patch_size),
                ),
            ]
        )
        dim_count += past_target.shape[-1]
        prediction_mask.extend(
            [
                torch.zeros(
                    batch_shape
                    + (self.context_token_length(patch_size) * past_target.shape[-1],),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    batch_shape
                    + (
                        self.prediction_token_length(patch_size)
                        * past_target.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )

        if feat_dynamic_real is not None:
            if observed_feat_dynamic_real is None:
                raise ValueError(
                    "observed_feat_dynamic_real must be provided if feat_dynamic_real is provided"
                )

            target.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., : self.hparams.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., self.hparams.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                ]
            )
            observed_mask.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., : self.hparams.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., self.hparams.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                ]
            )
            sample_id.extend(
                [
                    repeat(
                        reduce(
                            (
                                self._patched_seq_pad(
                                    patch_size, past_is_pad, -1, left=True
                                )
                                == 0
                            ).int(),
                            "... (seq patch) -> ... seq",
                            "max",
                            patch=patch_size,
                        ),
                        "... seq -> ... (dim seq)",
                        dim=feat_dynamic_real.shape[-1],
                    ),
                    torch.ones(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )
            time_id.extend(
                [past_seq_id] * feat_dynamic_real.shape[-1]
                + [future_seq_id] * feat_dynamic_real.shape[-1]
            )
            variate_id.extend(
                [
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                        past=self.context_token_length(patch_size),
                    ),
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                        future=self.prediction_token_length(patch_size),
                    ),
                ]
            )
            dim_count += feat_dynamic_real.shape[-1]
            prediction_mask.extend(
                [
                    torch.zeros(
                        batch_shape
                        + (
                            self.context_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.zeros(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )

        if past_feat_dynamic_real is not None:
            if past_observed_feat_dynamic_real is None:
                raise ValueError(
                    "past_observed_feat_dynamic_real must be provided if past_feat_dynamic_real is provided"
                )
            target.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                )
            )
            observed_mask.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                )
            )
            sample_id.append(
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_feat_dynamic_real.shape[-1],
                )
            )
            time_id.extend([past_seq_id] * past_feat_dynamic_real.shape[-1])
            variate_id.append(
                repeat(
                    torch.arange(past_feat_dynamic_real.shape[-1], device=device)
                    + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                )
            )
            dim_count += past_feat_dynamic_real.shape[-1]
            prediction_mask.append(
                torch.zeros(
                    batch_shape
                    + (
                        self.context_token_length(patch_size)
                        * past_feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                )
            )

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        prediction_mask = torch.cat(prediction_mask, dim=-1)
        return (target, observed_mask, sample_id, time_id, variate_id, prediction_mask)

    # Note: have not tested on multivariate data
    def _format_preds(
        self,
        num_quantiles: int,
        patch_size: int,
        preds: Float[torch.Tensor, "batch combine_seq patch"],
        target_dim: int,
    ) -> Float[torch.Tensor, "batch num_quantiles future_time *tgt"]:
        start = target_dim * self.context_token_length(patch_size)
        end = start + target_dim * self.prediction_token_length(patch_size)
        preds = preds[..., start:end, :num_quantiles, :patch_size]
        preds = rearrange(
            preds,
            "... (dim seq) num_quantiles patch -> ... num_quantiles (seq patch) dim",
            dim=target_dim,
        )[..., : self.hparams.prediction_length, :]
        return preds.squeeze(-1)

    @staticmethod
    def _expand_cache(cache: dict, factor: int) -> dict:
        """Broadcast a context-only cache along a new branch axis [B, ...] -> [B, F, ...]."""

        def exp(t: torch.Tensor) -> torch.Tensor:
            return repeat(t, "b ... -> b f ...", f=factor)

        return {
            "layer_kvs": [(exp(k), exp(v)) for (k, v) in cache["layer_kvs"]],
            "kv_var_id": exp(cache["kv_var_id"]),
            "kv_time_id": exp(cache["kv_time_id"]),
            "kv_sample_id": exp(cache["kv_sample_id"]),
            "variate_loc": exp(cache["variate_loc"]),
            "variate_scale": exp(cache["variate_scale"]),
            "ctx_len": cache["ctx_len"],
        }

    def get_default_transform(self) -> Transformation:
        transform = AsNumpyArray(
            field="target",
            expected_ndim=1 if self.hparams.target_dim == 1 else 2,
            dtype=np.float32,
        )
        if self.hparams.target_dim == 1:
            transform += AddObservedValuesIndicator(
                target_field="target",
                output_field="observed_target",
                imputation_method=CausalMeanValueImputation(),
                dtype=bool,
            )
            transform += ExpandDimArray(field="target", axis=0)
            transform += ExpandDimArray(field="observed_target", axis=0)
        else:
            transform += AddObservedValuesIndicator(
                target_field="target",
                output_field="observed_target",
                dtype=bool,
            )

        if self.hparams.feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(
                field="feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            )
            transform += AddObservedValuesIndicator(
                target_field="feat_dynamic_real",
                output_field="observed_feat_dynamic_real",
                dtype=bool,
            )

        if self.hparams.past_feat_dynamic_real_dim > 0:
            transform += AsNumpyArray(
                field="past_feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            )
            transform += AddObservedValuesIndicator(
                target_field="past_feat_dynamic_real",
                output_field="past_observed_feat_dynamic_real",
                dtype=bool,
            )
        return transform
