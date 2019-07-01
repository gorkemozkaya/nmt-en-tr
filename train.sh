PROBLEM=translate_en_tr
MODEL=transformer
HPARAMS=transformer_base

STORAGE_BUCKET=gs://gorkemozkaya

DATA_DIR=$STORAGE_BUCKET/data_v2
TMP_DIR=/tmp/tmp
TRAIN_DIR=$STORAGE_BUCKET/train_v2
USR_DIR=/Users/gorkemozkaya/Projects/NMT/nmt-en-tr/t2t/problems/nmt-en-tr

mkdir -p $TMP_DIR 

t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --generate_data \
  --data_dir=/tmp/t2t_data \
  --output_dir=/tmp/t2t_output \
  --problem=sort_words_according_to_length_random \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --train_steps=250000 \
  --eval_steps=25000
