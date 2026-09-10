#!/bin/bash
# Cross-check every Gym config path referenced by the Ultra stage YAMLs against
# the file listing of the ultra container image.
LIST=/tmp/ultra_sqsh_list.txt
CFG_DIR=examples/nemo_gym/nemotron-3-ultra

for stage in "$CFG_DIR"/*.yaml; do
  name=$(basename "$stage")
  refs=$(grep -oE '(resources_servers|responses_api_models|responses_api_agents)/[A-Za-z0-9_./-]+\.yaml' "$stage" | sort -u)
  missing=""
  total=0
  for r in $refs; do
    total=$((total + 1))
    if ! grep -q "Gym/${r}$" "$LIST"; then
      missing+="      MISSING: $r"$'\n'
    fi
  done
  printf '%-22s %2d refs' "$name" "$total"
  if [[ -n "$missing" ]]; then
    echo "  -> GAPS"
    printf '%s' "$missing"
  else
    echo "  -> all present"
  fi
done
