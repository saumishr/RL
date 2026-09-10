#!/bin/bash
# Cohort-retry-identity unit tests, run in-container on a CPU node.
#
# sbatch rather than srun: an agent shell teardown kills an srun step and leaves
# the allocation held with no step in it.
#
# Three prior attempts all died on:
#   ImportError: cannot import name '_console_main' from '_pytest.config'
#                (unknown location)
# "unknown location" means _pytest.config resolved to a namespace package, i.e.
# the parent's __path__ came back empty -- a partially-present tree, not a
# missing dependency. `--group test` did NOT fix it, so the remaining suspect is
# the pyproject.toml/uv.lock bind mounts: they come from our branch, the image's
# venv was built from the image's own lock, and every run logged
# "Uninstalled 1 package / Installed 1 package" -- uv reconciling our lock
# against a venv it did not build. This run drops those two mounts so uv sees
# the image's own metadata, and prints the state of _pytest either way so a
# fourth failure is diagnostic rather than another guess.
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=cohort-tests
#SBATCH --output=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra/cohort_tests.%j.log

set -euo pipefail

U=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra
T="$U/rlvr-allfeatures"

# Only the source trees are mounted. pyproject.toml and uv.lock are the image's.
srun --container-image="$U/containers/rl-gym.allfeatures-v2.sqsh" \
  --container-mounts=/lustre:/lustre,/scratch:/scratch,"$T/nemo_rl":/opt/nemo-rl/nemo_rl,"$T/tests":/opt/nemo-rl/tests \
  bash -lc '
    cd /opt/nemo-rl
    export UV_FROZEN=1 UV_HTTP_TIMEOUT=120 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

    echo "=========== DIAGNOSTICS ==========="
    P=/opt/nemo_rl_venv/bin/python
    echo "--- python: $($P -V 2>&1) ---"
    SP=$($P -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])" 2>&1)
    echo "--- site-packages: $SP ---"
    echo "--- _pytest tree ---"
    ls -la "$SP/_pytest/__init__.py" 2>&1 || echo "  NO _pytest/__init__.py"
    ls -la "$SP/_pytest/config/__init__.py" 2>&1 || echo "  NO _pytest/config/__init__.py"
    echo "--- import probe ---"
    $P -c "import _pytest, _pytest.config as c; print(\"_pytest.__path__ =\", _pytest.__path__); print(\"config file =\", getattr(c, \"__file__\", None))" 2>&1 | head -5
    echo "--- pytest dist-info present? ---"
    ls -d "$SP"/pytest-*.dist-info 2>&1 || echo "  no pytest dist-info"
    echo "==================================="

    echo "### attempt A: uv run --group test pytest"
    if uv run --group test pytest -q -p no:cacheprovider \
        tests/unit/experience/test_rollout_manager.py \
        tests/unit/experience/test_rollout_redispatch.py \
        tests/unit/experience/test_rollouts.py; then
      echo "RESULT=PASS_A"; exit 0
    fi

    echo "### attempt B: uv run --group test python -m pytest"
    if uv run --group test python -m pytest -q -p no:cacheprovider \
        tests/unit/experience/test_rollout_manager.py \
        tests/unit/experience/test_rollout_redispatch.py \
        tests/unit/experience/test_rollouts.py; then
      echo "RESULT=PASS_B"; exit 0
    fi

    echo "RESULT=FAIL_BOTH"
    exit 1
  '
