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

"""Unit tests for the fully-parallel checkpoint load patch.

Megatron Bridge constructs ``FullyParallelLoadStrategyWrapper`` with two
positional arguments, so the configured process group, exchange algorithm and
``per_rank_object_load`` never reach it. ``_patch_fully_parallel_load``
reinstates them; these tests pin the argument rewriting, since it edits a
positional argument list in place and a mistake there would silently swap the
base strategy for a process group.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def restore_wrapper():
    """Undo the global patch the module applies to mcore's wrapper class."""
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelLoadStrategyWrapper,
    )

    from nemo_rl.models.megatron import setup as setup_mod

    original_init = FullyParallelLoadStrategyWrapper.__init__
    setup_mod._fully_parallel_load_patched = False
    yield setup_mod
    FullyParallelLoadStrategyWrapper.__init__ = original_init
    setup_mod._fully_parallel_load_patched = False


def _checkpoint_cfg(group="ep_dp", algo="broadcast"):
    return SimpleNamespace(
        ckpt_fully_parallel_load_process_group=group,
        ckpt_fully_parallel_load_exchange_algo=algo,
    )


def _build_as_bridge_does():
    """Replicate Bridge's call: two positional arguments, nothing else."""
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelLoadStrategyWrapper,
    )

    return FullyParallelLoadStrategyWrapper("base-strategy", "dp-cp-group")


@pytest.mark.mcore
def test_ep_dp_overrides_the_hardcoded_group(restore_wrapper, monkeypatch):
    setup_mod = restore_wrapper
    monkeypatch.setattr(
        setup_mod, "_expert_data_parallel_replica_group", lambda: "ep-dp-group"
    )

    setup_mod._patch_fully_parallel_load(_checkpoint_cfg(group="ep_dp"))
    wrapper = _build_as_bridge_does()

    assert wrapper.parallelization_group == "ep-dp-group"
    # The group is rewritten positionally, so guard the neighbouring argument.
    assert wrapper.base_strategy == "base-strategy"
    assert wrapper.per_rank_object_load is True
    assert wrapper.exchange_algo == "broadcast"


@pytest.mark.mcore
def test_dp_leaves_bridges_group_untouched(restore_wrapper, monkeypatch):
    setup_mod = restore_wrapper

    def fail():
        raise AssertionError("expert group resolved for a 'dp' configuration")

    monkeypatch.setattr(setup_mod, "_expert_data_parallel_replica_group", fail)

    setup_mod._patch_fully_parallel_load(_checkpoint_cfg(group="dp"))
    wrapper = _build_as_bridge_does()

    assert wrapper.parallelization_group == "dp-cp-group"
    # 'dp' still needs the object-load fix; only the group is left alone.
    assert wrapper.per_rank_object_load is True


@pytest.mark.mcore
def test_per_rank_object_load_env_opt_out(restore_wrapper, monkeypatch):
    setup_mod = restore_wrapper
    monkeypatch.setenv("NRL_MEGATRON_PER_RANK_OBJECT_LOAD", "0")
    monkeypatch.setattr(
        setup_mod, "_expert_data_parallel_replica_group", lambda: "ep-dp-group"
    )

    setup_mod._patch_fully_parallel_load(_checkpoint_cfg(group="ep_dp"))
    wrapper = _build_as_bridge_does()

    assert wrapper.per_rank_object_load is False
    assert wrapper.parallelization_group == "ep-dp-group"


@pytest.mark.mcore
def test_unknown_process_group_raises(restore_wrapper):
    setup_mod = restore_wrapper

    with pytest.raises(ValueError, match="ckpt_fully_parallel_load_process_group"):
        setup_mod._patch_fully_parallel_load(_checkpoint_cfg(group="tp"))


@pytest.mark.mcore
def test_progress_dict_preserves_mapping_semantics():
    """mcore indexes and sizes this mapping as well as iterating it."""
    from nemo_rl.models.megatron.setup import _ProgressReportingDict

    source = {(index, 0, 0): index % 8 for index in range(250)}
    wrapped = _ProgressReportingDict(source, "rank 0:", 100)

    assert len(wrapped) == 250
    assert wrapped[(5, 0, 0)] == 5
    assert (7, 0, 0) in wrapped
    assert dict(wrapped) == source
    # The exchange relies on a deterministic order, so iteration must not
    # reorder or drop shards.
    assert [shard for shard, _ in wrapped.items()] == list(source)


@pytest.mark.mcore
def test_patch_is_idempotent(restore_wrapper, monkeypatch):
    setup_mod = restore_wrapper
    monkeypatch.setattr(
        setup_mod, "_expert_data_parallel_replica_group", lambda: "ep-dp-group"
    )

    cfg = _checkpoint_cfg(group="ep_dp")
    setup_mod._patch_fully_parallel_load(cfg)
    # The policy and reference-model loads both call this.
    setup_mod._patch_fully_parallel_load(cfg)
    wrapper = _build_as_bridge_does()

    assert wrapper.parallelization_group == "ep-dp-group"
    assert wrapper.base_strategy == "base-strategy"
