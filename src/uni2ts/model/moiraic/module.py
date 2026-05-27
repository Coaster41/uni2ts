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

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Bool, Float, Int
from torch import nn

from uni2ts.common.torch_util import packed_causal_attention_mask
from uni2ts.module.norm import RMSNorm
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedStdScaler
from uni2ts.module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from uni2ts.module.transformer import TransformerEncoder
from uni2ts.module.ts_embed import ResidualBlock


class MoiraicModule(
    nn.Module,
    PyTorchModelHubMixin,
):
    """
    Contains components of Moirai, to ensure implementation is identical across models.
    Subclasses huggingface_hub.PyTorchModelHubMixin to support loading from HuggingFace Hub.
    """

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
    ):
        """
        :param d_model: model hidden dimensions
        :param num_layers: number of transformer layers
        :param patch_size: patch size
        :param max_seq_len: maximum sequence length for inputs
        :param attn_dropout_p: dropout probability for attention layers
        :param dropout_p: dropout probability for all other layers
        :param scaling: whether to apply scaling (standardization)
        :param num_quantiles: number of quantile levels
        :param min_scale: minimum scale for std scaler
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_predict_token = num_predict_token
        self.max_seq_len = max_seq_len
        self.scaling = scaling
        self.quantile_levels = quantile_levels
        self.num_quantiles = len(quantile_levels)

        self.scaler = PackedStdScaler(minimum_scale=min_scale) if scaling else PackedNOPScaler()
        self.in_proj = ResidualBlock(
            input_dims=patch_size * 2,
            hidden_dims=d_model,
            output_dims=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=d_ff,
        )
        self.out_proj = ResidualBlock(
            input_dims=d_model,
            hidden_dims=d_model,
            output_dims=num_predict_token * self.num_quantiles * patch_size,
        )
        self.get_reprs = False

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        training_mode: Bool = True,
        past_cache: Optional[dict] = None,
        return_cache: bool = False,
        return_attn_weights: bool = False,
    ):
        """
        Defines the forward pass of MoiraiDecoderModule.
        This method expects processed inputs.

        1. Apply scaling to observations
        2. Project from observations to representations
        3. Replace prediction window with learnable mask
        4. Apply transformer layers
        5. Project from representations to distribution parameters
        6. Return distribution object

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
        :param training_mode: whether to use training mode (inference mode)
        :return: predictive distribution
        """
        if past_cache is None:
            # ------------------------------------------------------------------
            # Prefill path
            # ------------------------------------------------------------------
            loc, scale = self.scaler(
                target,
                observed_mask * ~prediction_mask.unsqueeze(-1),
                sample_id,
                variate_id,
            )
            scaled_target = (target - loc) / scale
            input_tokens = torch.cat(
                [scaled_target, observed_mask.to(torch.float32)], dim=-1
            )
            reprs = self.in_proj(input_tokens)

            attn_mask = packed_causal_attention_mask(sample_id, time_id)
            if return_cache:
                reprs, layer_kvs = self.encoder(
                    reprs, attn_mask, time_id=time_id, var_id=variate_id,
                    return_kvs=True,
                )
            elif return_attn_weights:
                reprs, all_attn_weights = self.encoder(
                    reprs, attn_mask, time_id=time_id, var_id=variate_id,
                    return_attn_weights=True,
                )
            else:
                reprs = self.encoder(
                    reprs, attn_mask, time_id=time_id, var_id=variate_id,
                )

            if self.get_reprs:
                preds = self.out_proj(reprs)
                result = (reprs, torch.cat((loc, scale), dim=-1))
            else:
                preds = self.out_proj(reprs)
                if training_mode:
                    result = (preds, scaled_target)
                else:
                    result = preds * scale + loc

            if return_cache:
                cache = self._build_context_cache(
                    layer_kvs=layer_kvs,
                    sample_id=sample_id,
                    time_id=time_id,
                    variate_id=variate_id,
                    prediction_mask=prediction_mask,
                    loc=loc,
                    scale=scale,
                )
                return result, cache
            if return_attn_weights:
                return result, all_attn_weights
            return result

        # ----------------------------------------------------------------------
        # Decode path — `target`, `observed_mask`, ids, `prediction_mask` cover
        # only the NEW tokens (caller is responsible for slicing).
        # ----------------------------------------------------------------------
        # 1) Look up loc/scale per new token using its variate_id.
        var_idx = variate_id.unsqueeze(-1)  # [..., new_len, 1]
        new_loc = past_cache["variate_loc"].gather(-2, var_idx)     # [..., new_len, 1]
        new_scale = past_cache["variate_scale"].gather(-2, var_idx) # [..., new_len, 1]

        # 2) Scale + project new tokens.
        scaled_target = (target - new_loc) / new_scale
        input_tokens = torch.cat(
            [scaled_target, observed_mask.to(torch.float32)], dim=-1
        )
        reprs = self.in_proj(input_tokens)

        # 3) Build attention mask spanning [cached_ctx ⊕ new] keys.
        full_sample_id = torch.cat([past_cache["kv_sample_id"], sample_id], dim=-1)
        full_time_id = torch.cat([past_cache["kv_time_id"], time_id], dim=-1)
        new_len = target.shape[-2]
        decode_attn_mask = packed_causal_attention_mask(
            full_sample_id, full_time_id
        )[..., -new_len:, :]

        # 4) Encoder over new tokens with cached context K/V.
        reprs = self.encoder(
            reprs,
            decode_attn_mask,
            time_id=time_id,
            var_id=variate_id,
            past_kvs=past_cache["layer_kvs"],
            past_kv_var_id=past_cache["kv_var_id"],
            past_kv_time_id=past_cache["kv_time_id"],
            return_kvs=False,  # context cache is invariant; nothing to update
        )

        # 5) Output projection on new tokens only.
        if self.get_reprs:
            preds = self.out_proj(reprs)
            result = (reprs, torch.cat((new_loc, new_scale), dim=-1))
        else:
            preds = self.out_proj(reprs)
            if training_mode:
                result = (preds, scaled_target)
            else:
                result = preds * new_scale + new_loc

        if return_cache:
            # Pass through the same context cache for chained AR calls.
            return result, past_cache
        return result


    def _build_context_cache(
        self,
        layer_kvs,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
        loc,
        scale,
    ) -> dict:
        """
        Filter the prefill cache down to context positions only and build
        a per-variate (loc, scale) lookup so that decode can rescale new
        tokens without re-running the scaler.

        Assumes the moiraic forecast layout: context positions form a
        contiguous prefix, prediction positions a contiguous suffix.
        """
        ctx_mask = ~prediction_mask  # [*B, S]; True where token is context
        # The forecast layout guarantees a contiguous context prefix; assert
        # so a future caller that violates it fails loudly here rather than
        # silently corrupting the cache.
        ctx_lens = ctx_mask.sum(dim=-1)
        ctx_len = int(ctx_lens.flatten()[0].item())
        assert (ctx_lens == ctx_len).all(), (
            "MoiraicModule cache assumes a uniform, contiguous context prefix; "
            f"got per-row context lengths {ctx_lens.tolist()}"
        )

        # Slice K/V along seq dim. layer_kvs[i] = (k, v) of shape
        # [..., group, hpg, S, head_dim]; we keep [..., :ctx_len, :].
        ctx_layer_kvs = [
            (k[..., :ctx_len, :].contiguous(), v[..., :ctx_len, :].contiguous())
            for (k, v) in layer_kvs
        ]
        ctx_var_id = variate_id[..., :ctx_len]
        ctx_time_id = time_id[..., :ctx_len]
        ctx_sample_id = sample_id[..., :ctx_len]

        # Build per-variate loc/scale lookup tables: shape [*B, num_variates, 1].
        # PackedStdScaler emits the same loc/scale for every position sharing a
        # (sample_id, variate_id) group; for a single-sample sequence (the
        # forecast case) this collapses to a per-variate table.
        num_variates = int(variate_id.max().item()) + 1
        batch_shape = variate_id.shape[:-1]
        variate_loc = torch.zeros(
            *batch_shape, num_variates, 1, dtype=loc.dtype, device=loc.device
        )
        variate_scale = torch.ones(
            *batch_shape, num_variates, 1, dtype=scale.dtype, device=scale.device
        )

        # Nondeterministic behavior breaks code:
        # variate_loc.scatter_(-2, ctx_var_id.unsqueeze(-1), loc[..., :ctx_len, :])
        # variate_scale.scatter_(-2, ctx_var_id.unsqueeze(-1), scale[..., :ctx_len, :])

        # Replacement:
        for v in range(num_variates):
            is_pred_v = prediction_mask & (variate_id == v)        # [*B, S]
            if not is_pred_v.any():
                continue
            first_v = is_pred_v & (is_pred_v.int().cumsum(-1) == 1)  # first True per row
            b_idx, s_idx = first_v.nonzero(as_tuple=True)
            variate_loc  [b_idx, v] = loc  [b_idx, s_idx]
            variate_scale[b_idx, v] = scale[b_idx, s_idx]

        return {
            "layer_kvs": ctx_layer_kvs,
            "kv_var_id": ctx_var_id,
            "kv_time_id": ctx_time_id,
            "kv_sample_id": ctx_sample_id,
            "variate_loc": variate_loc,
            "variate_scale": variate_scale,
            "ctx_len": ctx_len,
        }
