#!/bin/bash
#$ -m bae
#$ -M jporubci@nd.edu
#$ -N ArcIris
#$ -q gpu
#$ -l gpu=4
#$ -l h="qa-a10*|qa-rtx6k*"
#$ -o "/afs/crc.nd.edu/user/j/jporubci/Private/Fall 2024/ArcIris/stdout"
#$ -e "/afs/crc.nd.edu/user/j/jporubci/Private/Fall 2024/ArcIris/stderr"

FLUSH_TIMEOUT_SECONDS=900
fsync -d "$FLUSH_TIMEOUT_SECONDS" "$SGE_STDERR_PATH" &
fsync -d "$FLUSH_TIMEOUT_SECONDS" "$SGE_STDOUT_PATH" &

cd "/afs/crc.nd.edu/user/j/jporubci/Private/Fall 2024/ArcIris"
source "/afs/crc.nd.edu/user/j/jporubci/Private/Fall 2024/ArcIris/.venv/bin/activate"

python main.py \
  --batch_size 128 \
  --cuda \
  --cudnn \
  --debug \
  --distance_type euclidean \
  --finetune \
  --image_dir "/afs/crc/group/cvrl/czajka/gbir2/aczajka/BXGRID/iris_segmented_SegNet" \
  --img_uid_map "/afs/crc.nd.edu/user/j/jporubci/Private/Fall 2024/ArcIris/img_to_uid_map.json" \
  --log_batch 16 \
  --log_txt \
  --lr 0.00001 \
  --margin 2.0 \
  --mode train \
  --model_type convnext_small \
  --multi_gpu \
  --num_epochs 2048 \
  --num_workers 1 \
  --polar \
  --stem_width 64 \
  --tag polar
