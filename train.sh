export TPU_NAME=gorkemozkaya
#export TPU_NAME="projects/gorkem-tpu/locations/us-central1-a/nodes/gorkemozkaya"

PROBLEM=translate_en_tr
MODEL=transformer
HPARAMS=transformer_tpu

STORAGE_BUCKET=gs://gorkem-tpu

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
  --cloud_tpu_name=$TPU_NAME \
  --keep_checkpoint_max=20
