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

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from jaxtyping import Bool, Float

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin, ApplyFuncMixin
from uni2ts.common.typing import UnivarTimeSeries
from einops import rearrange


def _observed_patch_mean(
    arr: Float[np.ndarray, "var *time"],
    observed_mask: Bool[np.ndarray, "var *time"],
) -> Float[np.ndarray, "var ..."]:
    """Per-variate mean over observed positions, broadcastable back onto ``arr``.

    Returns a finite ``(var, 1, ...)`` array (0.0 where a variate has no observed
    position) so masked entries can be filled with a value consistent with the
    model's missing-data contract instead of NaN. The extra trailing 1-dims match
    ``arr.ndim`` so the result broadcasts directly against ``arr``.
    """
    var = arr.shape[0]
    obs = observed_mask.astype(arr.dtype)
    flat_sum = (arr * obs).reshape(var, -1).sum(axis=1)
    flat_cnt = obs.reshape(var, -1).sum(axis=1)
    fill = np.where(flat_cnt > 0, flat_sum / np.maximum(flat_cnt, 1.0), 0.0)
    return fill.reshape((var,) + (1,) * (arr.ndim - 1))


@dataclass
class ContextPatchMasking(ApplyFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    prediction_mask_field: str = "prediction_mask"
    observed_mask_field: str = "observed_mask"
    mask_ratio: float = 0.0

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.apply_func(
            self._add_context_patch_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry
    
    # Does not generate the same mask if there are multiple fields
    def _add_context_patch_mask(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr: Float[np.ndarray, "var time"] = data_entry[field]
        patch_size: int = data_entry["patch_size"]
        prediction_mask: Bool[np.ndarray, "var time"] = data_entry[
            self.prediction_mask_field
        ]
        observed_mask: Bool[np.ndarray, "var time"] = data_entry[
            self.observed_mask_field
        ][field]
        context_length = (~prediction_mask[0]).sum()  # same across all variates
        num_context_patches = context_length // patch_size

        if num_context_patches > 0 and self.mask_ratio > 0:
            num_mask = round(num_context_patches * self.mask_ratio)
            mask_indices = np.random.choice(
                num_context_patches, size=num_mask, replace=False
            )

            # Fill masked positions with a finite per-variate observed mean (not
            # NaN) so the (value, observed_mask=False) missing-data contract holds
            # and the scaler/network stay finite. See _observed_patch_mean.
            fill = _observed_patch_mean(arr[:, :context_length], observed_mask[:, :context_length])
            # Assume context is left aligned (skip next line)
            # context_start = context_length % patch_size
            for idx in mask_indices:
                patch_start = idx * patch_size
                patch_end = patch_start + patch_size
                arr[:, patch_start:patch_end] = fill
                observed_mask[:, patch_start:patch_end] = False


@dataclass
class ContiguousPatchMasking(ApplyFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    prediction_mask_field: str = "prediction_mask"
    observed_mask_field: str = "observed_mask"
    c_mask_max: int = 4      # TiRex Appendix D.3 default
    p_mask_max: float = 0.5  # TiRex Appendix D.3 default

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.apply_func(
            self._add_cpm_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _add_cpm_mask(self, data_entry: dict[str, Any], field: str) -> None:
        arr = data_entry[field]                                       # (var, total_patches, patch_size)
        prediction_mask = data_entry[self.prediction_mask_field]
        observed_mask = data_entry[self.observed_mask_field][field]   # (var, context_patches, patch_size)

        context_patches = int((~prediction_mask[0]).sum())            # already in patch units post-Patchify
        c_mask = np.random.randint(1, self.c_mask_max + 1)           # U(1, c_mask_max)
        p_mask = np.random.uniform(0.0, self.p_mask_max)             # U(0, p_mask_max)
        n_slots = context_patches // c_mask

        if n_slots > 0 and p_mask > 0:
            slot_mask = np.random.binomial(1, p_mask, size=n_slots).astype(bool)
            patch_mask = np.repeat(slot_mask, c_mask)                # (n_slots * c_mask,) patch units
            t = patch_mask.shape[0]
            expanded = patch_mask[np.newaxis, :, np.newaxis]         # (1, t, 1) broadcasts over var, patch_size
            # Masked patches follow the model's missing-data contract: a *finite*
            # value plus observed_mask=False. Writing np.nan here would poison the
            # scaler (nan * 0 = nan) and the network for the mask_inputs=false
            # decoder, where no mask_fill rescues these tokens. Fill with the
            # per-variate mean of observed context values (-> ~neutral scaled
            # input), falling back to 0.0 when a variate has no observed context.
            fill = _observed_patch_mean(
                arr[:, :context_patches], observed_mask[:, :context_patches]
            )
            arr[:, :t] = np.where(expanded, fill, arr[:, :t])
            observed_mask[:, :t] = np.where(expanded, False, observed_mask[:, :t])


@dataclass
class ContiguousPatchPrediction(Transformation):
    """CPM variant that treats masked context patches as prediction targets.

    Sets prediction_mask=True for CPM-selected context patches so the loss
    is computed on them directly (same gradient path as MaskedPrediction).
    Unlike ContiguousPatchMasking, observed_mask is untouched — the values
    are valid; they are hidden from the model via mask_inputs (encoder mode).
    Can be used alongside or instead of MaskedPrediction.
    """
    prediction_mask_field: str = "prediction_mask"
    c_mask_max: int = 4
    p_mask_max: float = 0.5

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        prediction_mask = data_entry[self.prediction_mask_field]
        context_patches = int((~prediction_mask[0]).sum())

        c_mask = np.random.randint(1, self.c_mask_max + 1)
        p_mask = np.random.uniform(0.0, self.p_mask_max)
        n_slots = context_patches // c_mask

        if n_slots > 0 and p_mask > 0:
            slot_mask = np.random.binomial(1, p_mask, size=n_slots).astype(bool)
            patch_mask = np.repeat(slot_mask, c_mask)        # (n_slots * c_mask,)
            t = patch_mask.shape[0]
            prediction_mask[:, :t] |= patch_mask[np.newaxis, :]

        return data_entry


@dataclass
class MaskedPrediction(MapFuncMixin, CheckArrNDimMixin, Transformation):
    min_mask_ratio: float
    max_mask_ratio: float
    target_field: str = "target"
    truncate_fields: tuple[str, ...] = tuple()
    optional_truncate_fields: tuple[str, ...] = tuple()
    prediction_mask_field: str = "prediction_mask"
    expected_ndim: int = 2

    def __post_init__(self):
        assert (
            self.min_mask_ratio <= self.max_mask_ratio
        ), "min_mask_ratio must be <= max_mask_ratio"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target = data_entry[self.target_field]
        prediction_mask = self._generate_prediction_mask(target)
        self.map_func(
            partial(self._truncate, mask=prediction_mask),  # noqa
            data_entry,
            self.truncate_fields,
            optional_fields=self.optional_truncate_fields,
        )
        data_entry[self.prediction_mask_field] = prediction_mask
        return data_entry

    def _generate_prediction_mask(
        self, target: Float[np.ndarray, "var time *feat"]
    ) -> Bool[np.ndarray, "var time"]:
        self.check_ndim("target", target, self.expected_ndim)
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)
        mask_ratio = np.random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        mask_length = max(1, round(time * mask_ratio))
        prediction_mask[:, -mask_length:] = True
        return prediction_mask

    def _truncate(
        self,
        data_entry: dict[str, Any],
        field: str,
        mask: np.ndarray,
    ) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        arr: np.ndarray | list[np.ndarray] | dict[str, np.ndarray] = data_entry[field]
        if isinstance(arr, list):
            return [self._truncate_arr(a, mask) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.truncate_fields or k in self.optional_truncate_fields:
                    arr[k] = self._truncate_arr(v, mask)
            return arr
        return self._truncate_arr(arr, mask)

    @staticmethod
    def _truncate_arr(
        arr: Float[np.ndarray, "var time *feat"], mask: Bool[np.ndarray, "var time"]
    ) -> Float[np.ndarray, "var time-mask_len *feat"]:
        return arr[:, ~mask[0]]


@dataclass
class ExtendMask(CheckArrNDimMixin, CollectFuncMixin, Transformation):
    fields: tuple[str, ...]
    mask_field: str
    optional_fields: tuple[str, ...] = tuple()
    expected_ndim: int = 2

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target_mask: np.ndarray = data_entry[self.mask_field]
        aux_target_mask: list[np.ndarray] = self.collect_func_list(
            self._generate_target_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        data_entry[self.mask_field] = [target_mask] + aux_target_mask
        return data_entry

    def _generate_target_mask(
        self, data_entry: dict[str, Any], field: str
    ) -> np.ndarray:
        arr: np.ndarray = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        field_target_mask = np.zeros((var, time), dtype=bool)
        return field_target_mask


@dataclass
class EvalMaskedPrediction(MapFuncMixin, CheckArrNDimMixin, Transformation):
    mask_length: int
    target_field: str = "target"
    truncate_fields: tuple[str, ...] = tuple()
    optional_truncate_fields: tuple[str, ...] = tuple()
    prediction_mask_field: str = "prediction_mask"
    expected_ndim: int = 2

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target = data_entry[self.target_field]
        prediction_mask = self._generate_prediction_mask(target)
        self.map_func(
            partial(self._truncate, mask=prediction_mask),  # noqa
            data_entry,
            self.truncate_fields,
            optional_fields=self.optional_truncate_fields,
        )
        data_entry[self.prediction_mask_field] = prediction_mask
        return data_entry

    def _generate_prediction_mask(
        self, target: Float[np.ndarray, "var time *feat"]
    ) -> Bool[np.ndarray, "var time"]:
        self.check_ndim("target", target, self.expected_ndim)
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)
        prediction_mask[:, -self.mask_length :] = True
        return prediction_mask

    def _truncate(
        self,
        data_entry: dict[str, Any],
        field: str,
        mask: np.ndarray,
    ) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        arr: np.ndarray | list[np.ndarray] | dict[str, np.ndarray] = data_entry[field]
        if isinstance(arr, list):
            return [self._truncate_arr(a, mask) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.truncate_fields or k in self.optional_truncate_fields:
                    arr[k] = self._truncate_arr(v, mask)
            return arr
        return self._truncate_arr(arr, mask)

    @staticmethod
    def _truncate_arr(
        arr: Float[np.ndarray, "var time *feat"], mask: Bool[np.ndarray, "var time"]
    ) -> Float[np.ndarray, "var time-mask_len *feat"]:
        return arr[:, ~mask[0]]
