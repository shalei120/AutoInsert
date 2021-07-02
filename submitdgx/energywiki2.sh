#!/bin/bash

#python3 -m pip install nltk
#python3 -m pip install bs4
#python3 -m pip install fairseq
#python3 -m pip install -r requirements.txt
# run the application
#echo "_AlanShore120" | sudo -S -k  apt-get update
#echo "_AlanShore120" | sudo -S -k  apt-get install build-essential --assume-yes
#echo "_AlanShore120" | sudo -S -k chmod 777 ./fairseq/
#cd ./fairseq/
#python3 -m pip install --editable ./
python3 -m pip install -r requirements.txt
#python3 main_mt.py -m transformer /-b 128  -d EN_DE -g 0 -layer 3 -s dgx #> energyLM-wiki2.out
#python3 setup.py build_ext --inplace
#cd ..
python3 textdata.py
TEXT=artifacts/fsdata
export PYTHONPATH=$PYTHONPATH:$PWD/fairseq
python3 fairseq/fairseq_cli/preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir artifacts/preprocessed/ \
    --workers 20
python3 main_mt.py -m transformer -b 64  -d DE_EN -g 0 -layer 6 -emb 512 -s dgx > slurm-aitf-$SLURM_JOB_ID.out
