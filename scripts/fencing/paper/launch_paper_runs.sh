#!/bin/bash
# Launch all paper training runs in the background (logs in output/paper/logs/).
#   GPU 1 (24 GB): flat baseline seed 0 + drills seed 0     GPU 0 (11 GB): drills seeds 1, then 2
# Each run needs ~8.7 GB. Check progress: tail -f output/paper/logs/*.log
cd /pub0/johnliao/SMPLOlympics
# GPU numbers below are nvidia-smi numbers; without this, CUDA orders GPUs fastest-first and
# CUDA_VISIBLE_DEVICES=1 lands on the 11 GB card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p output/paper/logs
FLAT_EPOCHS="${FLAT_EPOCHS:-40000}"; DRILL_EPOCHS="${DRILL_EPOCHS:-6000}"
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/fencing/paper/train_flat_baseline.sh 0 ${FLAT_EPOCHS} \
    > output/paper/logs/flat_s0.log 2>&1 &
echo "[launch] flat baseline s0 (GPU 1): pid $!"
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/fencing/paper/train_drills_warmstart.sh 0 ${DRILL_EPOCHS} \
    > output/paper/logs/drills_s0.log 2>&1 &
echo "[launch] drills s0 (GPU 1): pid $!"
CUDA_VISIBLE_DEVICES=0 nohup bash -c "bash scripts/fencing/paper/train_drills_warmstart.sh 1 ${DRILL_EPOCHS} > output/paper/logs/drills_s1.log 2>&1; \
    bash scripts/fencing/paper/train_drills_warmstart.sh 2 ${DRILL_EPOCHS} > output/paper/logs/drills_s2.log 2>&1" >/dev/null 2>&1 &
echo "[launch] drills s1 then s2 (GPU 0): pid $!"
