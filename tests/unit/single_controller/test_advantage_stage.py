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
"""The advantage stage's boundary: what it fetches and what it hands back."""

import pytest
import torch

from nemo_rl.algorithms.single_controller_utils.advantage_stage import (
    AdvantageOutcome,
    AdvantageRequest,
    AdvantageStageConfig,
    ImportanceSamplingDiagnosticsOptions,
    split_meta_by_prompt_group,
)
from nemo_rl.algorithms.single_controller_utils.config import AdvantageConfig
from nemo_rl.algorithms.single_controller_utils.utils import (
    AdvantagePartial,
    RewardPartial,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.utils.rpc_guard import assert_metadata_only


def _config(**overrides) -> AdvantageStageConfig:
    base = dict(
        advantage=AdvantageConfig(),
        algo=None,
        is_ppo=False,
        policy_logprobs_required=False,
        reference_logprobs_required=False,
        teacher_logprobs_required=False,
        message_level_advantage_penalties_enabled=False,
        importance_sampling_diagnostics=None,
    )
    base.update(overrides)
    return AdvantageStageConfig(**base)


class TestInputFields:
    def test_only_the_always_present_columns_by_default(self) -> None:
        adv = AdvantageConfig()
        assert _config().input_fields() == [
            adv.prompt_ids_field,
            adv.reward_field,
            adv.token_mask_field,
            adv.sample_mask_field,
            adv.mask_sample_field,
            adv.truncated_field,
        ]

    def test_policy_logprobs_pull_both_logprob_columns(self) -> None:
        adv = AdvantageConfig()
        fields = _config(policy_logprobs_required=True).input_fields()
        assert adv.policy_logprobs_field in fields
        assert adv.generation_logprobs_field in fields

    def test_ppo_pulls_values_and_penalties_pull_their_masks(self) -> None:
        adv = AdvantageConfig()
        fields = _config(
            is_ppo=True,
            message_level_advantage_penalties_enabled=True,
        ).input_fields()
        assert adv.values_field in fields
        assert adv.invalid_tool_call_mask_field in fields
        assert adv.malformed_thinking_mask_field in fields

    def test_repeated_batch_fields_do_not_duplicate_the_reward_column(self) -> None:
        adv = AdvantageConfig(repeated_batch_fields=["total_reward"])
        fields = _config(advantage=adv).input_fields()
        assert fields.count("total_reward") == 1


class TestRpcBoundaryStaysMetadataOnly:
    """The actor only helps if nothing cohort-sized crosses to it."""

    def test_request_carries_no_payload(self) -> None:
        request = AdvantageRequest(
            meta=KVBatchMeta(
                partition_id="rollout_data",
                task_name="train",
                sample_ids=["sample-0"],
                fields=["total_reward"],
            ),
            step=3,
            trainer_version=2,
        )
        assert_metadata_only(request)

    def test_outcome_carries_no_payload(self) -> None:
        outcome = AdvantageOutcome(
            meta=KVBatchMeta(
                partition_id="rollout_data",
                task_name="train",
                sample_ids=["sample-0"],
                fields=["advantages"],
            ),
            has_valid_training_tokens=True,
            num_mask_sample_filtered=1,
            reward_partial=RewardPartial.from_rows(torch.tensor([1.0, 3.0])),
            advantage_partial=AdvantagePartial.from_values(torch.tensor([0.5, 1.5])),
            seq_logprob_error_metrics={"max_seq_mult_prob_error": 1.25},
            opd_stat_sum=2.0,
            opd_stat_sumsq=2.5,
            opd_stat_count=2,
        )
        assert_metadata_only(outcome)

    def test_a_tensor_on_the_outcome_is_rejected(self) -> None:
        outcome = AdvantageOutcome(
            meta=KVBatchMeta(
                partition_id="rollout_data",
                task_name="train",
                sample_ids=["sample-0"],
                fields=["advantages"],
            ),
            has_valid_training_tokens=True,
            num_mask_sample_filtered=0,
            reward_partial=RewardPartial.from_rows(torch.tensor([1.0])),
            # The regression this guard exists for: handing back the tensor
            # instead of its reduction.
            advantage_partial=torch.zeros(4),  # type: ignore[arg-type]
        )
        with pytest.raises(TypeError, match="torch.Tensor"):
            assert_metadata_only(outcome)


class TestImportanceSamplingDiagnosticsOptions:
    def test_build_returns_an_empty_accumulator_with_the_options_applied(
        self,
    ) -> None:
        options = ImportanceSamplingDiagnosticsOptions(
            sequence_level_importance_ratios=True,
            truncated_importance_sampling_ratio=2.0,
            truncated_importance_sampling_ratio_min=0.5,
            truncated_importance_sampling_type="seq-mask-tis",
        )
        accumulator = options.build()
        assert accumulator.sequence_level_importance_ratios is True
        assert accumulator.truncated_importance_sampling_ratio == 2.0
        assert accumulator.truncated_importance_sampling_ratio_min == 0.5
        assert accumulator.truncated_importance_sampling_type == "seq-mask-tis"
        # Fresh per call, so it contributes nothing until something is recorded.
        assert accumulator.flush() == ({}, [])


def _grouped_meta(num_groups: int, per_group: int) -> KVBatchMeta:
    """Rows for ``num_groups`` contiguous prompt groups.

    Tags carry no ``dataset_source``, matching a dataset whose rows have no
    ``dataset`` key -- the case that made the tag-keyed splitter decline every
    time.
    """
    total = num_groups * per_group
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(total)],
        fields=["total_reward"],
        sequence_lengths=[8] * total,
        tags=[{"weight_version": 1} for _ in range(total)],
    )


class TestSplitMetaByPromptGroup:
    def test_shards_cover_every_row_in_original_order(self):
        meta = _grouped_meta(num_groups=8, per_group=4)
        shards = split_meta_by_prompt_group(meta, 4, 4)
        assert shards is not None and len(shards) == 4
        rejoined = [sid for shard in shards for sid in shard.sample_ids]
        assert rejoined == meta.sample_ids
        # Per-sample sidecars travel with the rows they describe.
        assert [t for shard in shards for t in shard.tags] == meta.tags
        assert [n for s in shards for n in s.sequence_lengths] == meta.sequence_lengths

    def test_every_shard_holds_whole_groups(self):
        meta = _grouped_meta(num_groups=8, per_group=4)
        shards = split_meta_by_prompt_group(meta, 3, 4)
        assert shards is not None
        assert all(len(shard.sample_ids) % 4 == 0 for shard in shards)

    def test_splits_without_any_dataset_source_tag(self):
        """Regression: the tag-keyed version declined here and measured nothing."""
        meta = _grouped_meta(num_groups=2048, per_group=16)
        shards = split_meta_by_prompt_group(meta, 8, 16)
        assert shards is not None
        assert [len(shard.sample_ids) for shard in shards] == [4096] * 8

    def test_never_returns_more_shards_than_requested(self):
        meta = _grouped_meta(num_groups=7, per_group=2)
        shards = split_meta_by_prompt_group(meta, 4, 2)
        assert shards is not None and len(shards) <= 4
        assert [s for sh in shards for s in sh.sample_ids] == meta.sample_ids

    def test_declines_when_groups_are_fewer_than_two(self):
        assert split_meta_by_prompt_group(_grouped_meta(1, 16), 8, 16) is None

    def test_declines_without_a_pool_to_spread_across(self):
        meta = _grouped_meta(num_groups=8, per_group=4)
        assert split_meta_by_prompt_group(meta, 1, 4) is None
        assert split_meta_by_prompt_group(meta, 0, 4) is None

    def test_declines_on_a_partial_group(self):
        """A row count that is not a whole multiple of the group size is not
        the layout this assumes, so it must not cut blind."""
        meta = _grouped_meta(num_groups=8, per_group=4)
        assert split_meta_by_prompt_group(meta.slice(0, 30), 4, 4) is None

    def test_declines_on_a_nonsense_group_size(self):
        meta = _grouped_meta(num_groups=8, per_group=4)
        assert split_meta_by_prompt_group(meta, 4, 0) is None
