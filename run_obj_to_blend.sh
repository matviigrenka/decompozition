#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat <<'EOF'
Usage:
  ./run_obj_to_blend.sh <input.obj> <checkpoint.pt> <blender_executable> <output.blend> [extra args...]

Example:
  ./run_obj_to_blend.sh \
    ./assets/chair.obj \
    ./checkpoints/model.pt \
    /c/Program\ Files/Blender\ Foundation/Blender\ 4.2/blender.exe \
    ./outputs/chair_parts.blend \
    --num-points 2048 --num-clusters 6 --use-normals --device cuda --vertex-groups --separate
EOF
  exit 1
fi

INPUT_OBJ="$1"
CHECKPOINT="$2"
BLENDER_BIN="$3"
OUTPUT_BLEND="$4"
shift 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/process_obj_to_blend.py" \
  --input "$INPUT_OBJ" \
  --checkpoint "$CHECKPOINT" \
  --blender "$BLENDER_BIN" \
  --output-blend "$OUTPUT_BLEND" \
  "$@"
