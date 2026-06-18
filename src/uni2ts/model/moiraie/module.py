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

"""Backward-compatible encoder preset of the unified :class:`MoiraiXModule`.

``MoiraieModule`` pins the masked-reconstruction encoder configuration
(``mask_inputs=True, predict_next=False``) and leaves ``causal`` configurable so
both the bidirectional and the causal-encoder checkpoints load unchanged. Because
``mask_inputs=True`` the ``mask_encoding`` parameter is created, matching the
original encoder checkpoints' state-dict keys.
"""

from uni2ts.model.moiraix.module import MoiraiXModule


class MoiraieModule(MoiraiXModule):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_layers: int,
        patch_size: int,
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
        num_predict_token: int = 1,
        quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_scale: float = 1e-5,
        causal: bool = False,
        train_scale_full_observed: bool = False,
        **kwargs,  # absorb/override any stale objective flags from old configs
    ):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            patch_size=patch_size,
            max_seq_len=max_seq_len,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            scaling=scaling,
            num_predict_token=num_predict_token,
            quantile_levels=quantile_levels,
            min_scale=min_scale,
            causal=causal,
            mask_inputs=True,
            predict_next=False,
            train_scale_full_observed=train_scale_full_observed,
        )
