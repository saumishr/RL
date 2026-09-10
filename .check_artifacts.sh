#!/bin/bash
# Verify HF checkpoint dirs are complete: config, tokenizer, and all safetensors
# shards referenced by the index are present.
check_model() {
  local label="$1" dir="$2"
  echo "----------------------------------------------------------------"
  printf '%-22s %s\n' "$label" "$dir"
  if [[ ! -d "$dir" ]]; then
    echo "  STATUS: MISSING (no such directory)"
    return
  fi
  if [[ ! -r "$dir" ]]; then
    echo "  STATUS: UNREADABLE"
    return
  fi

  # Some dirs wrap the real checkpoint one level down.
  local real="$dir"
  if [[ ! -f "$dir/config.json" ]]; then
    local sub
    sub=$(find "$dir" -maxdepth 2 -name config.json -printf '%h\n' 2>/dev/null | head -1)
    [[ -n "$sub" ]] && real="$sub"
  fi

  local idx="$real/model.safetensors.index.json"
  local n_shards want_shards
  n_shards=$(ls "$real"/*.safetensors 2>/dev/null | wc -l)
  if [[ -f "$idx" ]]; then
    want_shards=$(grep -oE '"model-[0-9]+-of-[0-9]+\.safetensors"' "$idx" 2>/dev/null | sort -u | wc -l)
  else
    want_shards="n/a"
  fi

  printf '  root:        %s\n' "$real"
  printf '  config.json: %s\n' "$([[ -f $real/config.json ]] && echo yes || echo NO)"
  printf '  tokenizer:   %s\n' "$(ls "$real" 2>/dev/null | grep -cE '^tokenizer|^vocab|^merges|\.jinja$') file(s)"
  printf '  safetensors: %s present / %s in index\n' "$n_shards" "$want_shards"
  printf '  size:        %s\n' "$(du -sh --apparent-size "$real" 2>/dev/null | cut -f1)"
  if [[ "$want_shards" != "n/a" && "$n_shards" -ne "$want_shards" ]]; then
    echo "  STATUS: INCOMPLETE — shard count mismatch"
  elif [[ "$n_shards" -eq 0 ]]; then
    echo "  STATUS: SUSPECT — no safetensors found"
  else
    echo "  STATUS: OK"
  fi
}

TK=/scratch/fsw/portfolios/nemotron/projects/nemotron_rl_algo/users/tkonuk/models
check_model "POLICY (SFT ckpt)"  "$TK/ultra-rl-prod_ultra_stage2sft300_fixlc-resumestep128-65k-step_152"
check_model "GENRM (reward)"     "$TK/qwen235b_principle_comparison_genrm_step1230"
check_model "NL2BASH judge FP8"  "$TK/Qwen3-235B-A22B-Instruct-2507-FP8"
check_model "SAFETY judge"       "$TK/Nemotron-Content-Safety-Reasoning-4B"
check_model "NL2BASH judge BF16" "/scratch/fsw/portfolios/nemotron/projects/nemotron_rl_algo/users/jiaqiz/models/Qwen3-235B-A22B-Instruct-2507"
check_model "NL2BASH bf16 (alt)" "/scratch/fsw/portfolios/nemotron/projects/nemotron_n3_post/users/igitman/hf_models/Qwen3-235B-A22B-Instruct-2507"
check_model "GENRM ultra-550B"   "/scratch/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/ilgeeh/hf_models/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM"
check_model "GENRM qwen235b-2603" "/scratch/fsw/portfolios/nemotron/projects/nemotron_rl_rm/users/ilgeeh/hf_models/Qwen3-Nemotron-235B-A22B-GenRM-2603"
echo "----------------------------------------------------------------"
