#!/usr/bin/env bash
export EN_TR_CORPUS_DIR=/Users/gorkemozkaya/Downloads/nmt_june_2019/en-tr.txt/

PROBLEM=translate_en_tr
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=/tmp/t2t/data
TMP_DIR=/tmp/tmp
TRAIN_DIR=/tmp/t2t/train
USR_DIR=/Users/gorkemozkaya/Projects/NMT/nmt-en-tr/nmt_en_tr/

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
