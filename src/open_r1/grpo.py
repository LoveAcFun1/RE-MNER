# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import random
import torch
from PIL import Image
import json

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "accuracy_reward_NER", "seg_NER_reward"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        if random.random() < 0.005:  # x% chance to write fully successful samples into a file
            print("\n==============")
            print(content)
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                ground_truth = sol

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth or ground_truth.replace('$','') == student_answer:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def energery_reward(ref_per_token_p, per_token_p, **kwargs):
    rewards = []
    for in_hidden_state, out_hidden_state in zip(ref_per_token_p, per_token_p):
        reward = 0.0
        try:
            abs_in = torch.abs(in_hidden_state)
            abs_out = torch.abs(out_hidden_state)
            ### 因为会变nan，所以加额外处理，避免下溢; update: 经过测试这一步对避免nan很重要
            abs_in = torch.clamp(abs_in, min=1e-6)
            abs_out = torch.clamp(abs_out, min=1e-6)
            
            ## 归一化，在embedding维度上看做一种概率分布;或者说每个token对应一个概率分布
            in_rela_value = abs_in / torch.sum(abs_in, dim=-1, keepdim=True) 
            out_rela_value = abs_out / torch.sum(abs_out, dim=-1, keepdim=True)
            
            
            in_log = torch.log(in_rela_value)
            in_entropy = - in_rela_value * torch.where(torch.isinf(in_log), 0., in_log)  # torch.special.entr(in_rela_value)
            ## 计算熵为什么不用pytorch的官方代码呢？官方代码有针对inf值的处理; update: 用这个报warning
            ## Warning: CAUTION: The operator 'aten::special_entr.out' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. (function npu_cpu_fallback)
            #in_entropy = torch.special.entr(in_rela_value)
            
            out_log = torch.log(out_rela_value)
            out_entropy = - out_rela_value * torch.where(torch.isinf(out_log), 0., out_log) # torch.special.entr(out_rela_value)
            #out_entropy = torch.special.entr(out_rela_value)
            
            entropy_diff = (out_entropy.sum(dim=-1) - in_entropy.sum(dim=-1)).mean()
            # print(entropy_diff)
            if entropy_diff < 0:
                reward = 1.0
            else:
                reward = 0.0
        except Exception:
            pass
    rewards.append(reward)
    return rewards   

def calculate_f1(dict1, dict2):
    # 计算交集（正确预测的数量）
    true_positives = sum(1 for k in dict1 if k in dict2 and dict1[k] == dict2[k])
    
    # dict1 中的总项数（预测总数）
    predicted_total = len(dict1)
    
    # dict2 中的总项数（实际总数）
    actual_total = len(dict2)

    if predicted_total == actual_total and actual_total == 0:
        return 1.0
    
    # 计算精确度和召回率
    precision = true_positives / predicted_total if predicted_total > 0 else 0
    recall = true_positives / actual_total if actual_total > 0 else 0
    
    # 计算 F1 分数
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score

def accuracy_reward_NER(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        if random.random() < 0.05:  # x% chance to write fully successful samples into a file
            print("\n==============")
            print(content)
        gold_parsed = sol
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            # answer_parsed = re.findall("<answer>(.*?)</answer>",content)
            label = json.loads(gold_parsed)
            answer_parsed = re.findall(r'\{[^{}]*\}',content)
            if len(answer_parsed)>0:
                try:
                    pred = json.loads(answer_parsed[0])
                    if pred is not None:
                        reward = calculate_f1(label, pred)
                    else:
                        reward = float(0)
                except:
                    reward = float(0)
            else:
                reward = float(0)
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

def accuracy_process_NER(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):

        gold_parsed = sol
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            # answer_parsed = re.findall("<answer>(.*?)</answer>",content)
            label = json.loads(gold_parsed)
            answer_parsed = re.findall(r'\{[^{}]*\}',content)
            if len(answer_parsed)>0:
                try:
                    pred = json.loads(answer_parsed[-1])
                    if pred is not None:
                        num = 0
                        for p in pred.keys():
                            if p in label:
                                num += 1
                            # else:
                            #     num -= 1
                        reward = 0.0 if len(label) == 0 else min(1.0, float(num/len(label)))
                        reward = 1.0 if len(label) == 0 and len(pred) == 0 else reward
                    else:
                        reward = float(0)
                except:
                    reward = float(0)
            else:
                reward = float(0)
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

def seg_NER_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):

        gold_parsed = sol
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            # answer_parsed = re.findall("<think>(.*?)</think>",content)
            answer_parsed = content
            num = 0.0
            # print(answer_parsed)
            if "image information:" in answer_parsed:
                num += 1.0
            if "text information:" in answer_parsed:
                num += 1.0
            if "multimodal information:" in answer_parsed:
                num += 1.0 
            # if "<think>" in answer_parsed:
            #     num += 1.0 
            # if "</think>" in answer_parsed:
            #     num += 1.0 
            # if "<answer>" in answer_parsed:
            #     num += 1.0 
            # if "</answer>" in answer_parsed:
                # num += 1.0 
            reward = num/3.0 

        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "energy": energery_reward,
    "accuracy_reward_NER": accuracy_reward_NER,
    "accuracy_process_NER":accuracy_process_NER,
    "seg_NER_reward":seg_NER_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    if script_args.dataset_name.find('MM-Eureka-Dataset') != -1:
        dataset = load_dataset('parquet', data_files=script_args.dataset_name)
        def add_image(example):
            # print(example)
            if example['image_urls'][0] != "blank_image.jpg":
                image_path = "/group/40064/johnbli/Code/Deepseek/open-r1-multimodal/datasets/MM-Eureka-Dataset/inspire/hdd/global_user/shaowenqi-shaowenqi/mengfanqing/OpenRLHF-InternVL/dataset/report_data/" + example['image_urls'][0]
            else:
                image_path = "/group/40064/johnbli/Code/Deepseek/open-r1-multimodal/datasets/MM-Eureka-Dataset/blank_image.jpg"
            return {
                "image": Image.open(image_path) 
            }
        dataset = dataset.map(add_image)
    elif script_args.dataset_name.find('json') != -1:
        dataset = load_dataset('json', data_files=script_args.dataset_name)
        def add_image(example):
            # print(example)
            image_path = example['image_urls']
            return {
                "image": Image.open(image_path) 
            }
        dataset = dataset.map(add_image)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags." #the image information in <image> </image> tags, 

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        # {"type": "text", "text": SYSTEM_PROMPT + example["problem"]},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    main(script_args, training_args, model_args)
