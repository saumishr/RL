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

"""Tests for SingleController initialization and pump lifecycle."""

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from tensordict import TensorDict

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.grpo import GRPOConfig, _initial_grpo_save_state
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    AsyncRLConfig,
    MasterConfig,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.rollout_timing import NemoGymRolloutTiming
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.timer import TimeoutChecker, Timer


class FakeWeightSynchronizer:
    pass


def _checkpointing_config(tmp_path) -> dict:
    """Minimal checkpointing block for actors built through __init__."""
    return {
        "enabled": False,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "metric_name": None,
        "higher_is_better": True,
        "keep_top_k": None,
        "save_period": 10,
        "save_optimizer": True,
        "checkpoint_must_save_by": None,
    }


def test_rejects_multiple_optimizer_steps_per_rl_step(monkeypatch) -> None:
    monkeypatch.setattr(single_controller, "Logger", lambda _: object())
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 4},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        async_rl=AsyncRLConfig(min_groups_for_streaming_train=1),
        logger={},
        env={},
    )
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=None,
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=None,
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        env_handles={},
        train_cluster=None,
        inference_cluster=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    with pytest.raises(
        ValueError,
        match=(
            r"num_prompts_per_step \* num_generations_per_prompt \(8\) "
            r"must equal policy.train_global_batch_size \(4\)"
        ),
    ):
        controller_cls(
            master_config=master_config,
            actor_args=actor_args,
            setup_timing_metrics=SetupTimingMetrics(),
        )


def test_logs_hyperparameters_and_concrete_weight_synchronizer(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(single_controller, "Logger", lambda _: logger)
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 8},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
        env={},
        # __init__ builds a CheckpointManager + TimeoutChecker from this block.
        checkpointing=_checkpointing_config(tmp_path),
    )
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=FakeWeightSynchronizer(),
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=None,
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        env_handles={},
        train_cluster=None,
        inference_cluster=None,
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
        setup_timing_metrics=SetupTimingMetrics(),
    )

    logger.log_hyperparams.assert_called_once_with(master_config.model_dump())
    output = capsys.readouterr().out
    assert "weight_sync=FakeWeightSynchronizer" in output
    assert "transport=stub" not in output


def test_logs_setup_timing_metrics(monkeypatch, tmp_path) -> None:
    """setup_timing_metrics is forwarded to Logger.log_metrics under timing/setup."""
    logger = MagicMock()
    monkeypatch.setattr(single_controller, "Logger", lambda _: logger)
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 8},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
        env={},
        # __init__ builds a CheckpointManager + TimeoutChecker from this block.
        checkpointing=_checkpointing_config(tmp_path),
    )
    setup_metrics = SetupTimingMetrics(
        generation_init_time_s=1.5, policy_init_time_s=2.5
    )
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=FakeWeightSynchronizer(),
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=None,
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        train_cluster=None,
        inference_cluster=None,
        # A real field of SingleControllerActorArgs. Read directly rather than via a
        # getattr default, so omitting it breaks here instead of silently degrading
        # watchdog.gym_subprocess_check into a no-op at runtime.
        env_handles={},
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
        setup_timing_metrics=setup_metrics,
    )

    logger.log_metrics.assert_called_once_with(
        setup_metrics.to_metrics_dict(), step=0, prefix="timing/setup"
    )


@pytest.mark.parametrize(
    ("recompute_kv_cache", "expected_invalidation_calls"),
    [(False, 0), (True, 1)],
)
def test_sync_weights_honors_recompute_kv_cache_config(
    recompute_kv_cache: bool,
    expected_invalidation_calls: int,
) -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig(
        recompute_kv_cache_after_weight_updates=recompute_kv_cache
    )
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(),
        requires_kv_scale_sync=False,
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._trainer_version = 3
    ctrl._inflight_by_group_id = {}
    # env={} -> _should_use_nemo_gym is False, so _sync_weights takes the native
    # abort path (empty registry -> no-op) instead of the gym gate.
    ctrl._master_config = SimpleNamespace(env={})

    asyncio.run(ctrl._sync_weights())

    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(kv_scales=None)
    assert ctrl._gen.invalidate_kv_cache.call_count == expected_invalidation_calls
    ctrl._rollout_manager.set_weight_version.assert_called_once_with(3)
    assert ctrl._rollout_permitted.is_set()


def test_sync_weights_calibrates_and_forwards_fp8_kv_scales() -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig()
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(),
        requires_kv_scale_sync=True,
    )
    ctrl._trainer = SimpleNamespace(
        calibrate_qkv_fp8_scales=MagicMock(return_value={"layers": {"layer.0": 0.5}})
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._trainer_version = 3
    ctrl._inflight_by_group_id = {}
    # env={} -> _should_use_nemo_gym is False, so _sync_weights takes the native
    # abort path (empty registry -> no-op) instead of the gym gate.
    ctrl._master_config = SimpleNamespace(env={})
    calibration_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    asyncio.run(ctrl._sync_weights(calibration_data=calibration_data))

    ctrl._trainer.calibrate_qkv_fp8_scales.assert_called_once_with(
        calibration_data,
        include_q=True,
    )
    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(
        kv_scales={"layer.0": 0.5}
    )


class _AdvantageDataPlane:
    def __init__(self, data: TensorDict) -> None:
        self._data = data
        self.selected_fields: list[str] | None = None
        self.written_fields: TensorDict | None = None

    def get_samples(self, *, select_fields, **kwargs):
        del kwargs
        self.selected_fields = list(select_fields)
        return self._data

    def put_samples(self, *, fields, **kwargs) -> None:
        del kwargs
        self.written_fields = fields


class _MaskRecordingAdvantageEstimator:
    def __init__(self) -> None:
        self.mask: torch.Tensor | None = None

    def compute_advantage(self, *, rewards, mask, **kwargs) -> torch.Tensor:
        del kwargs
        self.mask = mask.clone()
        return rewards.unsqueeze(-1).expand_as(mask).clone()


def test_advantage_stage_applies_seq_logprob_error_mask_before_streaming_train(
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_size, sequence_length = 4, 5
    generation_logprobs = torch.zeros(batch_size, sequence_length)
    # exp(abs(1 - 0)) > the configured threshold of 2, so only row 2
    # should be removed from the loss while the other rows remain trainable.
    generation_logprobs[2, 1:] = 1.0
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "prev_logprobs": torch.zeros(batch_size, sequence_length),
            "generation_logprobs": generation_logprobs,
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = _MaskRecordingAdvantageEstimator()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(seq_logprob_error_threshold=2.0)
    )
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    result_meta = asyncio.run(ctrl._advantage_stage(meta))
    capsys.readouterr()

    assert data_plane.selected_fields is not None
    assert "prev_logprobs" in data_plane.selected_fields
    assert "generation_logprobs" in data_plane.selected_fields
    assert data_plane.written_fields is not None
    assert torch.equal(
        data_plane.written_fields["sample_mask"],
        torch.tensor([1.0, 1.0, 0.0, 1.0]),
    )
    assert estimator.mask is not None
    assert estimator.mask[2].count_nonzero() == 0
    assert estimator.mask[[0, 1, 3]].all()
    metrics = ctrl._step_log_dict["seq_logprob_error_metrics"]
    assert len(metrics) == 1
    assert metrics[0]["num_masked_seqs_by_logprob_error"] == 1
    assert metrics[0]["max_seq_mult_prob_error"] == pytest.approx(math.e)
    assert metrics[0]["max_seq_mult_prob_error_after_mask"] == pytest.approx(1.0)
    assert "advantages" in (result_meta.fields or [])


class _EmptySampler:
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 0

    async def select(self, **kwargs):
        del kwargs
        return None, 0


class _OneThenEmptySampler(_EmptySampler):
    def __init__(self, meta: KVBatchMeta) -> None:
        self._meta: KVBatchMeta | None = meta

    async def select(self, **kwargs):
        del kwargs
        if self._meta is None:
            return None, 0
        meta = self._meta
        self._meta = None
        return meta, 1


class _EvictingSampler(_OneThenEmptySampler):
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 2

    async def select(self, **kwargs):
        meta, num_groups = await super().select(**kwargs)
        return meta, 2 if num_groups else 0


class _ChunkedSampler(_EmptySampler):
    """Assembles one step out of several single-group chunks, then goes empty.

    This is the shape the streaming path actually produces and the reason
    ``keep_train_buffers`` exists: every chunk after the first runs against an
    already-open train step.
    """

    def __init__(self, meta: KVBatchMeta, chunks: int) -> None:
        self._meta = meta
        self._remaining = chunks

    async def select(self, **kwargs):
        del kwargs
        if self._remaining == 0:
            return None, 0
        self._remaining -= 1
        return self._meta, 1


class _EmptyBuffer:
    def __len__(self) -> int:
        return 0


class _NoOpTrainer:
    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        del keep_train_buffers

    def prepare_for_training(self) -> None:
        pass

    def begin_train_step(self, loss_fn) -> None:
        del loss_fn

    def train_microbatches_from_meta(self, meta: KVBatchMeta) -> None:
        del meta

    def finish_train_step(self) -> dict:
        return {}


class _LpRecordingTrainer(_NoOpTrainer):
    """Records the ``keep_train_buffers`` flag the pump passes on each chunk."""

    def __init__(self) -> None:
        self.keep_train_buffers_calls: list[bool] = []

    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        self.keep_train_buffers_calls.append(keep_train_buffers)

    def get_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        del meta


class _NoOpDataPlane:
    def clear_samples(self, **kwargs) -> None:
        del kwargs


def _train_pump_controller(*, sampler) -> object:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            max_num_steps=1,
        ),
        # The pump's step epilogue reads the save triggers even when saving
        # is disabled.
        checkpointing={"enabled": False, "save_period": 10},
    )
    ctrl._async_cfg = SimpleNamespace(
        min_groups_for_streaming_train=1,
        rollout_failure=SimpleNamespace(min_step_batch_fraction=0.9),
    )
    ctrl._consumed_samples = 0
    ctrl._total_valid_tokens = 0
    ctrl._timeout = TimeoutChecker(timeout=None, fit_last_save_time=True)
    ctrl._timeout.start_iterations()
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._advantage_estimator = None
    ctrl._partition_id = "rollout_data"
    ctrl._sampler = sampler
    ctrl._buffer = _EmptyBuffer()
    # Read only by the select-stall warning, which reports what the pump is
    # waiting on.
    ctrl._inflight_rollouts = 0
    # stats.skipped is what the select-stall warning reports; the empty timing
    # pool is what the step-close postprocess-share report no-ops on.
    ctrl._rollout_manager = SimpleNamespace(
        stats=SimpleNamespace(skipped=0),
        rollout_timing=NemoGymRolloutTiming(),
    )
    ctrl._buffer_capacity = asyncio.Semaphore(2)
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._rollout_exhausted.set()
    ctrl._trainer = _NoOpTrainer()
    ctrl._gen = SimpleNamespace(
        requires_kv_scale_sync=False,
        # Interface defaults: no samples collected, so the step-close saturation
        # report no-ops, which is every backend but vLLM with its metrics logger on.
        get_logger_metrics=lambda: {},
        clear_logger_metrics=lambda: None,
    )
    ctrl._loss_fn = None
    ctrl._dp_client = _NoOpDataPlane()
    ctrl._timer = Timer()
    ctrl._trainer_version = 0
    ctrl._train_steps = 0
    ctrl._batch_shortfall = {}
    ctrl._batch_replacements = {}
    ctrl._batch_promotions = {}
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
    }
    return ctrl


def test_train_pump_stops_after_rollout_exhaustion_and_buffer_drain() -> None:
    ctrl = _train_pump_controller(sampler=_EmptySampler())

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 0


def test_train_pump_fails_if_rollout_exhausts_during_partial_step() -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_OneThenEmptySampler(meta))

    with pytest.raises(
        RuntimeError,
        match=(
            r"rollout exhausted before a complete training step was assembled: "
            r"dispatched 1/2 prompt groups"
        ),
    ):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))


async def _spin_train_pump(ctrl, seconds: float) -> None:
    """Run the pump against a sampler that never selects, then stop it.

    The pump only returns once a step closes or rollout is exhausted, and this is
    the case where neither happens.
    """
    task = asyncio.ensure_future(ctrl._train_pump())
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


class TestSelectStallWarning:
    """The stall the watchdog cannot see.

    Its idle timer resets on every commit, so a fleet that keeps rolling out into a
    step that never fills reads as healthy there while this pump spins in silence.
    """

    def _stalling_controller(self, monkeypatch, *, warn_after: float = 0.01):
        monkeypatch.setattr(single_controller, "_SELECT_STALL_WARN_SECONDS", warn_after)
        ctrl = _train_pump_controller(sampler=_EmptySampler())
        # Not exhausted: the pump waits rather than returning, which is the wedge.
        ctrl._rollout_exhausted.clear()
        return ctrl

    def test_a_pump_that_selects_nothing_says_so(self, capsys, monkeypatch) -> None:
        ctrl = self._stalling_controller(monkeypatch)
        ctrl._inflight_rollouts = 5
        ctrl._rollout_manager.stats.skipped = 3

        asyncio.run(_spin_train_pump(ctrl, 0.05))

        out = capsys.readouterr().out
        assert "train_pump selected no group" in out
        # The four numbers that separate a short-batch wedge from starvation at
        # the concurrency gate.
        assert "0/2 groups" in out
        assert "0 buffered" in out
        assert "5 rollouts in flight" in out
        assert "3 prompts skipped" in out

    def test_the_threshold_escalates_instead_of_printing_every_tick(
        self, capsys, monkeypatch
    ) -> None:
        """The pump retries every 5ms; one line per retry would bury the log.

        The k-th warning waits k thresholds, so over a window of 5 thresholds this
        prints about 5 lines where the pump retried about 20 times.
        """
        ctrl = self._stalling_controller(monkeypatch, warn_after=0.02)

        asyncio.run(_spin_train_pump(ctrl, 0.1))

        warnings = capsys.readouterr().out.count("train_pump selected no group")
        assert 1 <= warnings <= 6, f"printed {warnings} warnings in 100ms"

    def test_a_pump_still_making_progress_stays_quiet(
        self, capsys, monkeypatch
    ) -> None:
        monkeypatch.setattr(single_controller, "_SELECT_STALL_WARN_SECONDS", 30.0)
        ctrl = _train_pump_controller(sampler=_EmptySampler())
        ctrl._rollout_exhausted.clear()

        asyncio.run(_spin_train_pump(ctrl, 0.05))

        assert "train_pump selected no group" not in capsys.readouterr().out


def test_train_pump_logs_nonzero_stale_group_metrics(monkeypatch) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0", "sample-1"],
        fields=[],
        sequence_lengths=[1, 1],
        tags=[{"weight_version": 0}, {"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_EvictingSampler(meta))
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    ctrl._sync_weights.assert_awaited_once_with(calibration_data=None)
    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["evicted_stale_prompt_groups"] == 2
    assert train_metrics["aborted_stale_inflight_groups"] == 1


def test_train_pump_keeps_train_buffers_once_the_step_is_open(monkeypatch) -> None:
    """The logprob detour between chunks must not offload the trainer's grad
    buffers, because mcore's offload frees the gradients the earlier chunks of
    this step accumulated rather than copying them out.

    First chunk: no step open yet, nothing to preserve, so the offload is still
    worth taking. Every later chunk: step open, buffers must stay resident.
    """
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    # num_prompts_per_step is 2 in the harness, so two single-group chunks close
    # the step.
    ctrl = _train_pump_controller(sampler=_ChunkedSampler(meta, chunks=2))
    ctrl._policy_logprobs_required = True
    trainer = _LpRecordingTrainer()
    ctrl._trainer = trainer
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 1
    assert trainer.keep_train_buffers_calls == [False, True]
