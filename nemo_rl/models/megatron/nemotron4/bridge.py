# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Megatron -> HuggingFace weight bridge for Nemotron4 (NM4).

Megatron-Bridge ships providers for NM4 but no bridge, so a Megatron-trained
NM4 cannot be refit into vLLM: every vLLM refit transport exports through
``AutoBridge.export_hf_weights``, and only ``refit_transport=mcore`` (Megatron
to Megatron) avoids HF naming. This supplies the missing mapping.

Scope is the language model plus ``lm_head``. The vision tower and the
multimodal projector are deliberately absent because the NM4 vLLM fork maps
``model.vision_tower.`` and ``model.multi_modal_projector.`` to ``None`` and
is not ``SupportsMultiModal`` -- it discards them on load. The MTP tower is
absent for the same reason: the HF export has none.

Naming notes, each verified against a live 93B instantiation rather than the
checkpoint, because ``export_hf_weights`` walks ``named_parameters()``:

* ``sharded_state_dict()`` splits the GDP ``in_proj``/``conv1d`` into
  per-householder entries, but the live module keeps them fused at exactly the
  HF shapes, so both are plain renames.
* The live decoder numbers its 44 layers densely, matching
  ``text_config.layer_types`` one-to-one. The sparse 0..71 numbering, where
  each shortcut-MoE layer consumes two slots, exists only in the checkpoint.
  Layer wildcards therefore need no renumbering.
* Layer types never collide under wildcards because each sits on a distinct
  Megatron sub-path: a plain GDP layer is ``layers.N.mixer``, a shortcut-MoE
  one is ``layers.N.compute_layer.mixer``.
"""

from typing import Dict, Optional

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.experimental_nm4_llava_provider import NM4LlavaModel
from torch import nn

# ``wide_residual_layer._MIN_CONTROLLER_NUMEL``: stream logits are padded to
# this length so the distributed optimizer has a shard worth owning.
_LOGIT_PAD_NUMEL = 128


class WideResidualLogitMapping(ReplicatedMapping):
    """Trim a padded wide-residual stream logit to its active prefix.

    Megatron allocates every ``read_map``/``write_map``/``retention`` logit at
    ``_MIN_CONTROLLER_NUMEL`` and reads back ``param[:num_streams]``; the HF
    export stores only the active prefix (3 streams for NM4 Nano). Without
    this the exported tensor is 128 wide and vLLM rejects it.
    """

    @staticmethod
    def _num_streams(megatron_module: nn.Module) -> int:
        num_streams = getattr(megatron_module, "num_streams", None)
        if num_streams is None:
            raise AttributeError(
                "Expected a wide-residual module exposing 'num_streams' for "
                "logit trimming, got "
                f"{type(megatron_module).__name__}. The mapping is pointed at "
                "the wrong module."
            )
        return int(num_streams)

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        exported = super().megatron_to_hf(megatron_weights, megatron_module)
        if not exported:
            return exported
        num_streams = self._num_streams(megatron_module)
        return {name: tensor[:num_streams] for name, tensor in exported.items()}

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        num_streams = self._num_streams(megatron_module)
        if hf_weights.numel() != num_streams:
            raise ValueError(
                f"Expected {num_streams} stream logits, got {hf_weights.numel()}."
            )
        padded = hf_weights.new_zeros(max(_LOGIT_PAD_NUMEL, num_streams))
        padded[:num_streams] = hf_weights
        return super().hf_to_megatron(padded, megatron_module)


class GroupedExpertStackMapping(MegatronParamMapping[torch.Tensor]):
    """Stack per-expert Megatron weights into one HF 3D expert tensor.

    Transformer Engine's ``GroupedLinear`` exposes one parameter per local
    expert (``linear_fc1.weight0``, ``weight1``, ...), while the NM4 vLLM fork
    wants a single ``experts.up_proj``/``experts.down_proj`` of shape
    ``(num_experts, out, in)`` which it then enumerates by ``expert_id``.

    Each local expert parameter matches this mapping, but only the first one
    does the work: it stacks every local expert off the shared module and
    gathers the other expert-parallel ranks. The rest export nothing, since
    emitting the same HF tensor repeatedly would be wasted bandwidth.
    """

    def _local_hf_sharding(
        self, weight: torch.Tensor, megatron_module: nn.Module
    ) -> tuple[str, int | None, int]:
        return "replicated", None, 1

    @staticmethod
    def _local_expert_weights(megatron_module: nn.Module) -> list[torch.Tensor]:
        """Return this rank's expert weights in local expert order."""
        weights, index = [], 0
        while (weight := getattr(megatron_module, f"weight{index}", None)) is not None:
            weights.append(weight)
            index += 1
        if not weights:
            raise AttributeError(
                "Expected a GroupedLinear exposing 'weight0', 'weight1', ... on "
                f"{type(megatron_module).__name__}."
            )
        return weights

    def _is_first_local_expert(self) -> bool:
        return self.megatron_param.endswith("weight0")

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        # One export per group, driven by the first local expert.
        if not self._is_first_local_expert():
            return {}

        megatron_weights = self.broadcast_from_pp_rank(
            megatron_weights, cache_key=str(self.hf_param)
        )
        if megatron_weights is None or megatron_module is None:
            return {}

        local = torch.stack(
            [
                self.maybe_dequantize(weight)
                for weight in self._local_expert_weights(megatron_module)
            ]
        )

        from megatron.core import parallel_state

        ep_group = parallel_state.get_expert_model_parallel_group()
        ep_size = torch.distributed.get_world_size(group=ep_group)
        if ep_size == 1:
            return {str(self.hf_param): local}

        # Megatron hands each expert-parallel rank a contiguous block of
        # experts, so concatenating in rank order restores global expert ids.
        shards = [torch.empty_like(local) for _ in range(ep_size)]
        torch.distributed.all_gather(shards, local.contiguous(), group=ep_group)
        return {str(self.hf_param): torch.cat(shards, dim=0)}

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Return this rank's slice of the stacked HF expert tensor."""
        from megatron.core import parallel_state

        ep_group = parallel_state.get_expert_model_parallel_group()
        ep_size = torch.distributed.get_world_size(group=ep_group)
        ep_rank = torch.distributed.get_rank(group=ep_group)

        num_experts = hf_weights.shape[0]
        if num_experts % ep_size:
            raise ValueError(
                f"{num_experts} experts do not divide across {ep_size} "
                "expert-parallel ranks."
            )
        per_rank = num_experts // ep_size

        local_index = int(self.megatron_param.rsplit("weight", 1)[1])
        return hf_weights[ep_rank * per_rank + local_index].to(
            device=megatron_module.weight0.device
        )


def _gdp_mixer_mappings(megatron_mixer: str, hf_layer: str) -> list[MegatronParamMapping]:
    """Mappings shared by plain-GDP and shortcut-MoE-GDP layers."""
    return [
        # Transformer Engine fuses the pre-mixer norm into the projection.
        AutoMapping(
            megatron_param=f"{megatron_mixer}.in_proj.layer_norm_weight",
            hf_param=f"{hf_layer}.input_layernorm.weight",
        ),
        # Fused in the live module and already in HF's packed order
        # [z, V0..V2, K0..K2, Q, b0..b2, a], householder-major.
        AutoMapping(
            megatron_param=f"{megatron_mixer}.in_proj.weight",
            hf_param=f"{hf_layer}.mixer.in_proj.weight",
        ),
        AutoMapping(
            megatron_param=f"{megatron_mixer}.conv1d.weight",
            hf_param=f"{hf_layer}.mixer.conv1d.weight",
        ),
        AutoMapping(
            megatron_param=f"{megatron_mixer}.out_proj.weight",
            hf_param=f"{hf_layer}.mixer.out_proj.weight",
        ),
        AutoMapping(
            megatron_param=f"{megatron_mixer}.norm.weight",
            hf_param=f"{hf_layer}.mixer.norm.weight",
        ),
        ReplicatedMapping(
            megatron_param=f"{megatron_mixer}.A_log",
            hf_param=f"{hf_layer}.mixer.A_log",
        ),
        ReplicatedMapping(
            megatron_param=f"{megatron_mixer}.dt_bias",
            hf_param=f"{hf_layer}.mixer.dt_bias",
        ),
    ]


def _residual_mappings(megatron_prefix: str, hf_prefix: str) -> list[MegatronParamMapping]:
    """Read/write/retention logits for one wide-residual connection."""
    return [
        WideResidualLogitMapping(
            megatron_param=f"{megatron_prefix}.read_map.logit",
            hf_param=f"{hf_prefix}.read_logit",
        ),
        WideResidualLogitMapping(
            megatron_param=f"{megatron_prefix}.write_map.logit",
            hf_param=f"{hf_prefix}.write_logit",
        ),
        WideResidualLogitMapping(
            megatron_param=f"{megatron_prefix}.retention.retention_logit",
            hf_param=f"{hf_prefix}.retention_logit",
        ),
    ]


@MegatronModelBridge.register_bridge(
    source="Nemotron4ForConditionalGeneration",
    target=NM4LlavaModel,
    model_type="nemotron4",
)
class Nemotron4Bridge(MegatronModelBridge):
    """Weight bridge for NM4, export direction only.

    NM4 has no HF modeling code, so this is registered by architecture name
    rather than by class.
    """

    def provider_bridge(self, hf_pretrained):
        raise NotImplementedError(
            "NM4 Megatron models are built by ExperimentalNM4LlavaProvider from "
            "the checkpoint's own saved args, not from an HF config -- see that "
            "provider's docstring. This bridge exists only to name weights for "
            "refit, so point policy.model_name at an NM4 HF export and let the "
            "checkpoint supply the architecture."
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        layer = "language_model.decoder.layers.*"
        hf_layer = "model.language_model.layers.*"

        mappings: list[MegatronParamMapping] = [
            AutoMapping(
                megatron_param="language_model.embedding.word_embeddings.weight",
                hf_param="model.language_model.embed_tokens.weight",
            ),
            AutoMapping(
                megatron_param="language_model.decoder.final_norm.weight",
                hf_param="model.language_model.norm.weight",
            ),
            AutoMapping(
                megatron_param="language_model.output_layer.weight",
                hf_param="lm_head.weight",
            ),
            WideResidualLogitMapping(
                megatron_param="language_model.decoder.residual_stream_readout.exit_map.logit",
                hf_param="model.language_model.residual_readout.logit",
            ),
        ]

        # Plain GDP layers.
        mappings += _gdp_mixer_mappings(f"{layer}.mixer", hf_layer)
        mappings += _residual_mappings(
            f"{layer}.residual_connection", f"{hf_layer}.residual_connection"
        )

        # Plain MLP layers. NM4 uses squared ReLU, so the MLP is ungated and
        # linear_fc1 is a bare up-projection rather than a fused gate+up.
        mappings += [
            AutoMapping(
                megatron_param=f"{layer}.mlp.linear_fc1.layer_norm_weight",
                hf_param=f"{hf_layer}.input_layernorm.weight",
            ),
            AutoMapping(
                megatron_param=f"{layer}.mlp.linear_fc1.weight",
                hf_param=f"{hf_layer}.mlp.up_proj.weight",
            ),
            AutoMapping(
                megatron_param=f"{layer}.mlp.linear_fc2.weight",
                hf_param=f"{hf_layer}.mlp.down_proj.weight",
            ),
        ]
        mappings += _residual_mappings(
            f"{layer}.residual_connection_mlp", f"{hf_layer}.residual_connection"
        )

        # Shortcut-MoE layers: GDP mixer variant.
        mappings += _gdp_mixer_mappings(f"{layer}.compute_layer.mixer", hf_layer)
        mappings += _residual_mappings(
            f"{layer}.compute_layer.residual_connection",
            f"{hf_layer}.mixer_residual_connection",
        )

        # Shortcut-MoE layers: attention mixer variant. num_query_groups is 1,
        # so linear_qkv is a plain [q|k|v] concatenation with no per-group
        # interleaving for QKVMapping to undo.
        mappings += [
            AutoMapping(
                megatron_param=f"{layer}.compute_layer.self_attention.linear_qkv.layer_norm_weight",
                hf_param=f"{hf_layer}.input_layernorm.weight",
            ),
            QKVMapping(
                megatron_param=f"{layer}.compute_layer.self_attention.linear_qkv.weight",
                q=f"{hf_layer}.self_attn.q_proj.weight",
                k=f"{hf_layer}.self_attn.k_proj.weight",
                v=f"{hf_layer}.self_attn.v_proj.weight",
            ),
            AutoMapping(
                megatron_param=f"{layer}.compute_layer.self_attention.linear_proj.weight",
                hf_param=f"{hf_layer}.self_attn.o_proj.weight",
            ),
        ]
        mappings += _residual_mappings(
            f"{layer}.compute_layer.residual_connection_self_attn",
            f"{hf_layer}.mixer_residual_connection",
        )

        # The MoE block attached to both shortcut variants.
        mappings += [
            AutoMapping(
                megatron_param=f"{layer}.shortcut_pre_mlp_layernorm.weight",
                hf_param=f"{hf_layer}.shortcut_input_layernorm.weight",
            ),
            WideResidualLogitMapping(
                megatron_param=f"{layer}.shortcut_residual_read.read_map.logit",
                hf_param=f"{hf_layer}.shortcut_read.logit",
            ),
            AutoMapping(
                megatron_param=f"{layer}.moe_layer.pre_mlp_layernorm.weight",
                hf_param=f"{hf_layer}.pre_moe_layernorm.weight",
            ),
            ReplicatedMapping(
                megatron_param=f"{layer}.moe_layer.mlp.router.weight",
                hf_param=f"{hf_layer}.moe.gate.weight",
            ),
            # A persistent buffer rather than a parameter; export covers both.
            ReplicatedMapping(
                megatron_param=f"{layer}.moe_layer.mlp.router.expert_bias",
                hf_param=f"{hf_layer}.moe.gate.e_score_correction_bias",
            ),
            AutoMapping(
                megatron_param=f"{layer}.moe_layer.mlp.fc1_latent_proj.weight",
                hf_param=f"{hf_layer}.moe.fc1_latent_proj.weight",
            ),
            AutoMapping(
                megatron_param=f"{layer}.moe_layer.mlp.fc2_latent_proj.weight",
                hf_param=f"{hf_layer}.moe.fc2_latent_proj.weight",
            ),
            AutoMapping(
                megatron_param=f"{layer}.moe_layer.mlp.fc2_norm.weight",
                hf_param=f"{hf_layer}.moe.fc2_norm.weight",
            ),
            AutoMapping(
                megatron_param=f"{layer}.moe_layer.mlp.shared_experts.linear_fc1.weight",
                hf_param=f"{hf_layer}.moe.shared_experts.up_proj.weight",
            ),
            AutoMapping(
                megatron_param=f"{layer}.moe_layer.mlp.shared_experts.linear_fc2.weight",
                hf_param=f"{hf_layer}.moe.shared_experts.down_proj.weight",
            ),
            GroupedExpertStackMapping(
                megatron_param=f"{layer}.moe_layer.mlp.experts.linear_fc1.weight*",
                hf_param=f"{hf_layer}.moe.experts.up_proj",
            ),
            GroupedExpertStackMapping(
                megatron_param=f"{layer}.moe_layer.mlp.experts.linear_fc2.weight*",
                hf_param=f"{hf_layer}.moe.experts.down_proj",
            ),
        ]
        mappings += _residual_mappings(
            f"{layer}.moe_layer.residual_connection_mlp",
            f"{hf_layer}.moe_residual_connection",
        )

        return MegatronMappingRegistry(*mappings)
