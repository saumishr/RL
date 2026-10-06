# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare a long-context NeMo-Gym RLVR file for a short-context policy.

The Super 3.5 production RLVR set is built for 64k context and references
roughly thirty resource servers. NT4 Nano caps at 8192 positions, and the
NT4 recipe wires only the judge-driven subset of those servers. Neither
mismatch can be handled from config:

  * ``nemo_gym_data_processor`` returns a placeholder for text rows and never
    looks at ``max_seq_length``, because Gym assembles the real prompt
    server-side. An overlong row is therefore not skipped, it is generated
    against and silently truncated at the vLLM ``max_model_len``.
  * A row whose ``agent_ref`` names a resource server that is not in
    ``env.nemo_gym.config_paths`` fails at rollout rather than being ignored.

So both filters happen here, on the file. Retention is reported per reason and
per agent so the drop is auditable rather than assumed.

Example:
    python examples/nemo_gym/nt4-nano/filter_rlvr_by_length.py \
      --input  rl-v43-broad-falcon-baf8_noncommercial-resume.train.len64k.jsonl \
      --output rl-v43.nt4nano.len4k.jsonl \
      --tokenizer /path/to/nt4-nano-processor \
      --max-prompt-tokens 4096
"""

import argparse
import json
from collections import Counter

# The agent families served by the resource servers the NT4 recipe wires. Rows
# outside this set are dropped because nothing would answer them, not because
# they are uninteresting -- widening the recipe's config_paths is what widens
# this list.
DEFAULT_KEEP_PREFIXES = (
    "genrm_",
    "equivalence_llm_judge",
    "lc_judge",
    "lc_equivalence_rule",
    "math_with_judge",
    "instruction_following",
    "mcqa",
    "jailbreak",
)


def agent_name(row: dict) -> str:
    ref = row.get("agent_ref")
    if isinstance(ref, dict):
        return str(ref.get("name", ""))
    return str(ref or "")


def prompt_text(row: dict) -> str:
    messages = row.get("responses_create_params", {}).get("input", [])
    if isinstance(messages, str):
        return messages
    return "\n".join(
        str(m.get("content") or "") for m in messages if isinstance(m, dict)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer or processor directory for the target policy",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument(
        "--keep-agent-prefix",
        action="append",
        default=None,
        help="Repeatable. Defaults to the judge-driven families the NT4 recipe wires.",
    )
    parser.add_argument(
        "--drop-agent-prefix",
        action="append",
        default=None,
        help=(
            "Repeatable, applied after --keep-agent-prefix. Use this when the "
            "recipe wires nearly every server and it is the exceptions that are "
            "worth naming, which is clearer than restating thirty keep prefixes."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after writing this many rows. Useful for a smoke file.",
    )
    args = parser.parse_args()

    keep = tuple(args.keep_agent_prefix or DEFAULT_KEEP_PREFIXES)
    drop = tuple(args.drop_agent_prefix or ())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Cheap pre-screen so the tokenizer is not run on a 176k-character row.
    # Conservative on purpose: the observed ratio on this file is ~3.5
    # characters a token, so screening at 8 only discards rows that could not
    # fit even if every token were twice as long as anything seen here.
    char_budget = args.max_prompt_tokens * 8

    dropped: Counter[str] = Counter()
    kept_by_agent: Counter[str] = Counter()
    dropped_by_agent: Counter[str] = Counter()
    total = written = 0

    with (
        open(args.input, encoding="utf-8") as src,
        open(args.output, "w", encoding="utf-8") as dst,
    ):
        for line in src:
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped["unparseable"] += 1
                continue

            name = agent_name(row)
            if not name.startswith(keep) or (drop and name.startswith(drop)):
                dropped["unwired_agent"] += 1
                dropped_by_agent[name] += 1
                continue

            text = prompt_text(row)
            if len(text) > char_budget:
                dropped["too_long_prescreen"] += 1
                dropped_by_agent[name] += 1
                continue

            n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if n_tokens > args.max_prompt_tokens:
                dropped["too_long"] += 1
                dropped_by_agent[name] += 1
                continue

            dst.write(line if line.endswith("\n") else line + "\n")
            written += 1
            kept_by_agent[name] += 1
            if args.max_rows is not None and written >= args.max_rows:
                break

    print(f"read {total} rows, wrote {written} ({written / max(total, 1):.1%})")
    print("dropped:")
    for reason, count in dropped.most_common():
        print(f"  {count:8d}  {reason}")
    print("kept by agent:")
    for name, count in kept_by_agent.most_common():
        print(f"  {count:8d}  {name}")
    print("dropped by agent:")
    for name, count in dropped_by_agent.most_common():
        print(f"  {count:8d}  {name}")


if __name__ == "__main__":
    main()
