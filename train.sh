PROBLEM=translate_en_tr
MODEL=transformer
HPARAMS=transformer_base

STORAGE_BUCKET=gs://gorkemozkaya

DATA_DIR=$STORAGE_BUCKET/data_v2
TMP_DIR=/tmp/tmp
TRAIN_DIR=$STORAGE_BUCKET/train_v2
USR_DIR=/home/gorkemozkaya/nmt-en-tr/t2t/problems/nmt-en-tr

mkdir -p $TMP_DIR 

t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --output_dir=$TRAIN_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --train_steps=250000 \
  --eval_steps=25000 \
  --use_tpu=True \
  --tpu_name=gorkemozkaya \
  --keep_checkpoint_max=20
