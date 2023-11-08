OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi

BATCH_SIZE=6

mkdir -p $OUTPUT

deepspeed main.py \
  --deepspeed ./finetune/ds_config/ds_config_zero3.json \
  --model_name_or_path Nanbeige/Nanbeige-16b-base \
  --dataset_name data/ \
  --do_train \
  --max_length 4096 \
  --gradient_checkpointing true \
  --output_dir $OUTPUT \
  --overwrite_output_dir \
  --preprocess_num_workers 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --weight_decay 0.0001 \
  --bf16 True \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 20 \
  --logging_steps 10 \
  --gradient_accumulation_steps 4 \
  --report_to tensorboard \
  --tf32 True \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE
