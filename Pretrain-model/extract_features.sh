export BERT_BASE_DIR=~/model/protein-bert

python ./bert/extract_features.py \
  --input_file=$BERT_BASE_DIR/brct_varients.txt \
  --output_file=$BERT_BASE_DIR/feature/brca1.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab_1mer.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/brct_out/model.ckpt-200000 \
  --layers=-1 \
  --max_seq_length=256 \
  --batch_size=16
