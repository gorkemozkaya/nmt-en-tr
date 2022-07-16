export PYTHONPATH=$PYTHONPATH:/content/nmt-en-tr/models:/content/nmt-en/tr/datasets
python3 /content/nmt-en-tr/models/official/nlp/data/train_sentencepiece.py \
  --character_coverage=1.0 \
  --data_keys=en,tr \
  --tfds_name=blended_translate/tr-en \
  --tfds_split=train \
  --tfds_dir=gs://gorkem-tpu \
  --output_model_path=/content/sentencepiece_en_tr \
  --max_char=100000000
