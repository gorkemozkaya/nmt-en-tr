OUT_DIR=/Users/gorkemozkaya/Projects/NMT/storage_bucket/train_v2
MODEL=transformer
HPARAMS=transformer_tpu
PROBLEM=translate_en_tr

USR_DIR=/Users/gorkemozkaya/Projects/NMT/nmt-en-tr/t2t/problems/nmt-en-tr
DATA_DIR=/tmp/t2t/data

#mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$OUT_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_interactive=True