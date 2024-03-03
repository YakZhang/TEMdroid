
CUDA_VISIBLE_DEVICES=0 python main_cosine.py \
  --model_name_or_path bert-base-uncased \
  --do_train --train_file train_data \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --output_dir output_dir \
  --per_device_eval_batch_size=64 \
  --per_device_train_batch_size=64 \
  --overwrite_output \
  --save_strategy steps \
  --save_steps 400 \
  --do_eval --validation_file validation_data  \
  --do_predict --test_file test_data \
  --mask_index -1 \
  --mask_value -1 \






