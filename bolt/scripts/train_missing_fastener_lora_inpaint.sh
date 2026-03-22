#!/bin/bash
set -e
# Missing fastener LoRA for SD1.5 inpainting

pretrained_model="/path/to/sd-v1-5-inpainting.ckpt"
model_type="sd1.5"
parameterization=0

# 建议把两个 ROI 脚本导出的目录合并到同一个 train_data_dir 下
# 例如：
# train_data/
# ├── 10_missing_fastener/   <- close-up ROI
# └── 6_missing_fastener/    <- mid-context ROI
train_data_dir="./train_data"
reg_data_dir=""

network_module="networks.lora"
network_weights=""
network_dim=8
network_alpha=8

resolution="512,512"
batch_size=1
max_train_epoches=10
save_every_n_epochs=1

train_unet_only=1
train_text_encoder_only=0
stop_text_encoder_training=0

noise_offset="0.05"
keep_tokens=1
min_snr_gamma=5

lr="5e-5"
unet_lr="5e-5"
text_encoder_lr="1e-5"
lr_scheduler="cosine"
lr_warmup_steps=0
lr_restart_cycles=1

optimizer_type="AdamW8bit"

output_name="missing_fastener_inpaint_lora"
save_model_as="safetensors"

save_state=0
resume=""

min_bucket_reso=384
max_bucket_reso=768
persistent_data_loader_workers=1
clip_skip=2
multi_gpu=0
lowram=0

algo="lora"
conv_dim=4
conv_alpha=4
dropout="0"

use_wandb=0
wandb_api_key=""
log_tracker_name=""

export HF_HOME="huggingface"
export TF_CPP_MIN_LOG_LEVEL=3

extArgs=()
launchArgs=()
trainer_file="./scripts/stable/train_network.py"

if [ "$model_type" == "sd1.5" ]; then
  extArgs+=("--clip_skip=$clip_skip")
elif [ "$model_type" == "sd2.0" ]; then
  extArgs+=("--v2")
elif [ "$model_type" == "sdxl" ]; then
  trainer_file="./scripts/stable/sdxl_train_network.py"
elif [ "$model_type" == "flux" ]; then
  trainer_file="./scripts/dev/flux_train_network.py"
fi

if [[ $multi_gpu == 1 ]]; then
  launchArgs+=("--multi_gpu")
  launchArgs+=("--num_processes=2")
fi

if [[ $lowram == 1 ]]; then extArgs+=("--lowram"); fi
if [[ $parameterization == 1 ]]; then extArgs+=("--v_parameterization"); fi
if [[ $train_unet_only == 1 ]]; then extArgs+=("--network_train_unet_only"); fi
if [[ $train_text_encoder_only == 1 ]]; then extArgs+=("--network_train_text_encoder_only"); fi
if [[ -n "$network_weights" ]]; then extArgs+=("--network_weights=$network_weights"); fi
if [[ -n "$reg_data_dir" ]]; then extArgs+=("--reg_data_dir=$reg_data_dir"); fi
if [[ -n "$optimizer_type" ]]; then extArgs+=("--optimizer_type=$optimizer_type"); fi
if [[ "$optimizer_type" == "DAdaptation" ]]; then extArgs+=("--optimizer_args" "decouple=True"); fi
if [[ $save_state == 1 ]]; then extArgs+=("--save_state"); fi
if [[ -n "$resume" ]]; then extArgs+=("--resume=$resume"); fi
if [[ $persistent_data_loader_workers == 1 ]]; then extArgs+=("--persistent_data_loader_workers"); fi
if [[ $network_module == "lycoris.kohya" ]]; then
  extArgs+=("--network_args" "conv_dim=$conv_dim" "conv_alpha=$conv_alpha" "algo=$algo" "dropout=$dropout")
fi
if [[ $stop_text_encoder_training -ne 0 ]]; then extArgs+=("--stop_text_encoder_training=$stop_text_encoder_training"); fi
if [[ "$noise_offset" != "0" ]]; then extArgs+=("--noise_offset=$noise_offset"); fi
if [[ $min_snr_gamma -ne 0 ]]; then extArgs+=("--min_snr_gamma=$min_snr_gamma"); fi

if [[ $use_wandb == 1 ]]; then
  extArgs+=("--log_with=all")
  if [[ -n "$wandb_api_key" ]]; then extArgs+=("--wandb_api_key=$wandb_api_key"); fi
  if [[ -n "$log_tracker_name" ]]; then extArgs+=("--log_tracker_name=$log_tracker_name"); fi
else
  extArgs+=("--log_with=tensorboard")
fi

python -m accelerate.commands.launch ${launchArgs[@]} --num_cpu_threads_per_process=4 "$trainer_file" \
  --enable_bucket \
  --pretrained_model_name_or_path="$pretrained_model" \
  --train_data_dir="$train_data_dir" \
  --output_dir="./output" \
  --logging_dir="./logs" \
  --log_prefix="$output_name" \
  --resolution="$resolution" \
  --network_module="$network_module" \
  --max_train_epochs="$max_train_epoches" \
  --learning_rate="$lr" \
  --unet_lr="$unet_lr" \
  --text_encoder_lr="$text_encoder_lr" \
  --lr_scheduler="$lr_scheduler" \
  --lr_warmup_steps="$lr_warmup_steps" \
  --lr_scheduler_num_cycles="$lr_restart_cycles" \
  --network_dim="$network_dim" \
  --network_alpha="$network_alpha" \
  --output_name="$output_name" \
  --train_batch_size="$batch_size" \
  --save_every_n_epochs="$save_every_n_epochs" \
  --mixed_precision="fp16" \
  --save_precision="fp16" \
  --seed="1337" \
  --cache_latents \
  --prior_loss_weight=1 \
  --max_token_length=225 \
  --caption_extension=".txt" \
  --save_model_as="$save_model_as" \
  --min_bucket_reso="$min_bucket_reso" \
  --max_bucket_reso="$max_bucket_reso" \
  --keep_tokens="$keep_tokens" \
  --xformers --shuffle_caption \
  ${extArgs[@]}
