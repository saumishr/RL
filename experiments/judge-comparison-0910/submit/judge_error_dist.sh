#!/bin/bash
# Judge-model error distribution, discovered rather than assumed.
#
# Earlier passes here kept producing phantom counts, so this script does not
# grep for a preset list of error names. It extracts every *Error/*Exception
# token and every "NNN <phrase>" HTTP status actually present, then reports what
# it found. It also separates the three judge MODEL logs (the processes that talk
# to NVCF) from the judge AGENT logs (which call those models), because only the
# former can show an NVCF transport failure.
set -u

run_dir=$1
label=$2
G="$run_dir/runs/latest/logs/nemo_gym"
[[ -d "$G" ]] || G=$(ls -d "$run_dir"/runs/*/logs/nemo_gym 2>/dev/null | tail -1)

MODELS="genrm_model nl2bash_judge_model safety_judge_model"

echo "=============================================================="
echo "$label"
echo "  log dir: $G"
echo "=============================================================="

echo
echo "-- judge MODEL logs (the NVCF-facing processes) --"
for m in $MODELS; do
  f="$G/$m.log"
  if [[ ! -f "$f" ]]; then
    printf "  %-22s (absent)\n" "$m"
    continue
  fi
  # Exception class tokens, excluding the ones that are only ever substrings of
  # a longer name already counted.
  errs=$(grep -ohE '[A-Za-z]+(Error|Exception|Timeout)\b' "$f" 2>/dev/null | sort | uniq -c | sort -rn | tr '\n' ' ')
  # HTTP statuses only when followed by a status phrase, so ports and
  # millisecond fields in timestamps cannot match.
  http=$(grep -ohE '\b(4[0-9]{2}|5[0-9]{2}) [A-Z][A-Za-z-]+' "$f" 2>/dev/null | sort | uniq -c | sort -rn | tr '\n' ' ')
  printf "  %-22s size=%-9s errors=[%s] http=[%s]\n" \
    "$m" "$(stat -c%s "$f")" "${errs:-none}" "${http:-none}"
done

echo
echo "-- judge AGENT logs (callers of the above) --"
for f in "$G"/*judge*.log "$G"/*genrm*.log; do
  [[ -f "$f" ]] || continue
  b=$(basename "$f")
  case " $MODELS " in *" ${b%.log} "*) continue ;; esac
  errs=$(grep -ohE '[A-Za-z]+(Error|Exception|Timeout)\b' "$f" 2>/dev/null | sort | uniq -c | sort -rn | tr '\n' ' ')
  [[ -z "$errs" ]] && continue
  printf "  %-46s [%s]\n" "$b" "$errs"
done

echo
echo "-- every error token across ALL gym logs, ranked --"
grep -rohE '[A-Za-z]+(Error|Exception)\b' "$G"/*.log 2>/dev/null \
  | sort | uniq -c | sort -rn | head -12 | sed 's/^/  /'
echo "  (none)" && true
