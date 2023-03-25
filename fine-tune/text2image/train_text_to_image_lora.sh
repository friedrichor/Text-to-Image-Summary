export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_DISK_PATH="/datas/kfh/finetune_sd/text2image/datasets/Text2Image_example"
export OUTPUT_DIR="/datas/kfh/finetune_sd/text2image/txt2img-finetune-lora"
export VALID_PROMPT_DIR="/datas/kfh/finetune_sd/text2image/validation_prompts.txt"
export NEGATIVE_PROMPT_DIR="/datas/kfh/finetune_sd/text2image/validation_negative_prompts.txt"

accelerate launch --mixed_precision="fp16" /datas/kfh/finetune_sd/text2image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_from_disk \
  --dataset_disk_path=$DATASET_DISK_PATH \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --max_train_steps=500 \
  --gradient_accumulation_steps=8 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --use_8bit_adam \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=500 \
  --validation_prompts_dir=$VALID_PROMPT_DIR \
  --validation_negative_prompts_dir=$NEGATIVE_PROMPT_DIR \
  --num_validation_images=4 \
  --validation_epochs=50 \
  --report_to="tensorboard"