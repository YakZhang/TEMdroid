

# test
CUDA_VISIBLE_DEVICES=0 python main_cosine.py \
  --model_name_or_path bert-base-uncased \
  --resume_from_checkpoint model_path \
  --learning_rate 5e-5 \
  --num_train_epochs 15 \
  --output_dir $2 \
  --per_device_eval_batch_size=128 \
  --per_device_train_batch_size=128 \
  --overwrite_output \
  --save_strategy steps \
  --save_steps 1000 \
  --do_predict --test_file $1 \
  --mask_index -1 \
  --mask_value -1 \





