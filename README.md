# Reasoning-Enhanced Multimodal Named Entity Recognition

## Install
Mostly refer to fire-fly installation
1. Clone this repository and navigate to project folder

2. Install Package
```Shell
conda create -n RE-MNER python=3.10 -y
conda activate RE-MNER
pip install -e ".[dev]"
pip install -r requirements.txt
```

## Train
To train the models, follow these steps:

```Shell
torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12347 \
    src/open_r1/grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir {output_dir} \
    --model_name_or_path {model_path} \
    --dataset_name {your_dataset_path} \
    --max_prompt_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to tensorboard \
    --save_steps 100\
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 2359296 \
    --save_total_limit 8 \
    --num_train_epochs 2 \
    --num_generations 8 \
    --run_name $WANDB_RUN_NAME \
```
