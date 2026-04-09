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

            # Assume context is left aligned (skip next line)
            # context_start = context_length % patch_size
            for idx in mask_indices:
                patch_start = idx * patch_size
                patch_end = patch_start + patch_size
                arr[:, patch_start:patch_end] = np.nan
                observed_mask[:, patch_start:patch_end] = False


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
