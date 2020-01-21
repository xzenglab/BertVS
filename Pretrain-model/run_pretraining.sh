export BERT_BASE_DIR=~/model/protein-bert

python ./bert/run_pretraining.py \
  --input_file=$BERT_BASE_DIR/pretrain_output/brct.tfrecord \
  --output_dir=$BERT_BASE_DIR/brct_out \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=16 \
  --max_seq_length=256 \
  --max_predictions_per_seq=20 \
  --learning_rate=2e-5
