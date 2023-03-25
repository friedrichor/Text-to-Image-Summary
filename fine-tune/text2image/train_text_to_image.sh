export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_DISK_PATH="/datas/kfh/finetune_sd/text2image/datasets/Text2Image_example"
export OUTPUT_DIR="/datas/kfh/finetune_sd/text2image/txt2img-finetune"
export VALID_PROMPT_DIR="/datas/kfh/finetune_sd/text2image/validation_prompts.txt"
export NEGATIVE_PROMPT="disfigured, ugly, bad, immature"

accelerate launch --mixed_precision="fp16"  /datas/kfh/finetune_sd/text2image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_from_disk \
  --dataset_disk_path=$DATASET_DISK_PATH \
  --use_ema \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --max_train_steps=500 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=100 \
  --validation_prompts_dir=$VALID_PROMPT_DIR \
  --validation_negative_prompt=$NEGATIVE_PROMPT \
  --num_validation_images=4 \
  --validation_epochs=50 \
  --report_to="tensorboard"