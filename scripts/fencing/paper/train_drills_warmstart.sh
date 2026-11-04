#!/bin/bash
# Paper run, our method: fine-tune the drills policy under the final (v23) recipe, warm-started from
# the development checkpoint. One output dir per seed; all seeds start from the same weights.
#   bash scripts/fencing/paper/train_drills_warmstart.sh <seed> [epochs=6000] [init_checkpoint]
# Output: output/HumanoidIm/paper_drills_s<seed>/
cd /pub0/johnliao/SMPLOlympics
SEED="${1:?usage: $0 <seed> [epochs] [init_checkpoint]}"
EPOCHS="${2:-6000}"
INIT="${3:-output/HumanoidIm/fencing_drills_v23/Humanoid_00144000.pth}"
OUT=output/HumanoidIm/paper_drills_s${SEED}
[[ -f "${INIT}" ]] || { echo "[error] init checkpoint not found: ${INIT}"; exit 1; }
N=$(basename "${INIT}" .pth | sed 's/Humanoid_//'); N=$((10#$N))
mkdir -p "${OUT}"
if ! ls ${OUT}/Humanoid_[0-9]*.pth >/dev/null 2>&1; then
    cp "${INIT}" "${OUT}/"
    cp "${INIT%.pth}_op.pth" "${OUT}/"
    echo "[paper] warm start: ${INIT} -> ${OUT}"
fi
# v24 recipe = v23 + en-garde lead-in before half of the stationary lunges (lunging after holding
# en garde landed 18% vs 98% from spawn with v23). GUARD=0 to train the plain v23 recipe instead.
GUARD="${GUARD:-0.5}"
EXP_NAME_OVERRIDE=paper_drills_s${SEED} bash scripts/fencing/train_fencing_drills.sh C ${EPOCHS} \
    seed=${SEED} +env.lunge_guard_prob=${GUARD} '+env.lunge_guard_steps=[10,40]' "${@:4}"
