#!/bin/bash
# ps -ef | grep keepworking_v4 | awk '{print$2}' | xargs kill -9
export NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

GPUS="0,1,2,3,4,5,6,7"
ARNOLD_WORKER_GPU = 8
ARNOLD_WORKER_NUM = 1
ARNOLD_ID = 0
METIS_WORKER_0_HOST = 127.0.0.1
port_in_cmd = 12345

# 取 worker0 第一个 port
ports=($(echo $METIS_WORKER_0_PORT | tr ',' ' '))
port=${ports[0]}
port_in_cmd="$(echo "${METIS_WORKER_0_PORT:-2000}" | awk -F',' '{print $1}')"

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"
echo "master port in cmd: ${port_in_cmd}"

# export WANDB_BASE_URL=https://api.wandb.ai
# export WANDB_API_KEY="<PLACEHOLDER_WANDB_KEY_1>"
# wandb login $WANDB_API_KEY

# export WANDB_BASE_URL=https://api.wandb.ai
# export WANDB_PROJECT=vision-reasoning
# export WANDB_API_KEY="af66fb3345273759c1d8d9d3d6720e8f24caad6c"
export WANDB_RUN_NAME=Qwen2-VL-7B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
# wandb login $WANDB_API_KEY

# cd /home/tiger/multimodal-open-r1
# pip3 install vllm==0.6.6.post1
# pip3 install -e ".[dev]"
# pip3 install wandb==0.18.3



torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12347 \
    src/open_r1/grpo.py \
    --deepspeed /group/40064/johnbli/Code/Deepseek/open-r1-multimodal/local_scripts/zero3.json \
    --output_dir checkpoints/Qwen2-VL-7B-cold-0.4-continue-cold-segreward-all \
    --model_name_or_path /group/40064/johnbli/saved_models/MNER/Qwen2-VL-7B-cold-0.4 \
    --dataset_name /group/40064/johnbli/Data/GMNER/data/train_2015_2017_0.4.json \
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
    &> logs/Qwen2-VL-7B-cold-0.4-continue-cold-segreward-all.log


# nohup python /group/40064/johnbli/Code/keepworking_v4/run.py &