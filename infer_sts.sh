#!/usr/bin/env bash

MODELS=models
BERT_BASE_DIR=$MODELS/uncased_L-12_H-768_A-12
GLUE_DIR=glue_data
TRAINED_SCORER=sts_output
INFER_DIR=sts_infer

python run_reg.py \
  --task_name=STS-B \
  --do_predict=true \
  --data_dir=$GLUE_DIR/STS-B \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_SCORER \
  --max_seq_length=128 \
  --output_dir=$INFER_DIR
