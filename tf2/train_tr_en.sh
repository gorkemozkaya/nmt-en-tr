#train the tr -> en translator

export PYTHONPATH=$PYTHONPATH:/content/nmt-en-tr/models
export PYTHONPATH=/content/nmt-en-tr/datasets:$PYTHONPATH
export PARAMS=runtime.distribution_strategy=tpu
export PARAMS=$PARAMS,task.sentencepiece_model_path=gs://gorkem-tpu/tf2_model_tren_blended_v2/sentencepiece_en_tr.model
export PARAMS=$PARAMS,task.print_translations=true
export EXPERIMENT=transformer_tr_en_blended/base
export MODEL_DIR=gs://gorkem-tpu/tf2_model_tren_blended_v2
export TFDS_DATA_DIR=gs://gorkem-tpu
echo $PARAMS
python3 /content/nmt-en-tr/models/official/nlp/train.py \
  --experiment=${EXPERIMENT} \
  --mode=train_and_eval \
  --model_dir=${MODEL_DIR} \
  --tpu=${TPU_NAME} \
  --params_override=$PARAMS
