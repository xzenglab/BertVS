export BERT_BASE_DIR=~/model/protein-bert

python ./bert/create_pretraining_data.py \
  --input_file=$BERT_BASE_DIR/pfam_brca1.txt \
  --output_file=$BERT_BASE_DIR/pretrain_output/brct.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab_1mer.txt \
  --do_lower_case=True \
  --max_seq_length=256\
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
