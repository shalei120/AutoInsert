#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --job-name=AItf
#SBATCH --gres=gpu:1

module load cuda/9.2

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
echo $PWD
python3 -m pip install fairseq
# run the application
python3 textdata.py
TEXT=artifacts/fsdata
export PYTHONPATH=$PYTHONPATH:$PWD/fairseq
python3 fairseq/fairseq_cli/preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir artifacts/preprocessed/ \
    --workers 20
python3 main_mt.py -m transformer -b 64  -d DE_EN -g 0 -layer 6 -emb 512  > slurm-aitf-$SLURM_JOB_ID.out
