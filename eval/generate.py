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
import json

from PIL import Image


from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import numpy as np

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import copy

import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url


class GRPOTrainer_new(Qwen2VLGRPOTrainer):

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        label_txt = [x['solution'] for x in inputs]
        labels = self.processing_class(text = label_txt, return_tensors="pt",
                                        padding=True,
                                        padding_side="left",
                                        add_special_tokens=False).input_ids
        labels = labels.cuda()
        try:
        # if True:
            prompts = [x["prompt"] for x in inputs]
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            images = [x["image"] for x in inputs]
            prompt_inputs = self.processing_class(
                text=prompts_text,
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            prompt_inputs = {key: value.to('cuda') for key, value in prompt_inputs.items()}

            # print("input_ids: {}".format(prompt_inputs['input_ids'].shape))
            # print("attention_mask: {}".format(prompt_inputs['attention_mask'].shape))

            if self.max_prompt_length is not None:
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

            # Generate completions
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                # prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

                # Generate N times, each generate one with the temp_generation_config , stack the output_ids to prompt_completion_ids, pad the empty places with number 151613
                num_generations = self.generation_config.num_return_sequences
                temp_generation_config = copy.deepcopy(self.generation_config)
                temp_generation_config.num_return_sequences = 1

                all_completions = []

                for i in range(num_generations):  # -1 because we already have one generation
                    completion = unwrapped_model.generate(**prompt_inputs, generation_config=temp_generation_config)
                    all_completions.append(completion)

                if random.random() < 0.01:  # x% chance to write fully successful samples into a file
                    print("\n==============")
                    print(all_completions[-1])

                # Stack all completions and pad if needed
                max_length = max(completion.size(1) for completion in all_completions)
                padded_completions = []

                for completion in all_completions:
                    if completion.size(1) < max_length:
                        padding = torch.full(
                            (completion.size(0), max_length - completion.size(1)),
                            self.processing_class.tokenizer.pad_token_id,
                            dtype=completion.dtype,
                            device=completion.device,
                        )
                        padded_completion = torch.cat([completion, padding], dim=1)
                    else:
                        padded_completion = completion
                    padded_completions.append(padded_completion)

                # Stack all padded completions
                prompt_completion_ids = torch.cat(padded_completions, dim=0)

            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
        except:
            # print(Exception)
            print("BUG!!!!!!!!!!!!!")
            print(inputs)
            return None, None, None
        generated_tokens = completion_ids
        return None, generated_tokens, labels


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
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
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
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

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        # {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["question_en"])},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
            "solution":example["solution"]
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = GRPOTrainer_new

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_train_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    ##########
    # Evaluate
    ##########
    processing_class = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    pad_token_id = processing_class.tokenizer.pad_token_id
    processing_class.pad_token_id = pad_token_id
    processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
    print("*** Evaluate ***")
    predict_results = trainer.predict(trainer.eval_dataset, metric_key_prefix="predict")
    predict = np.where(predict_results.predictions<0, 0, predict_results.predictions)
    response = processing_class.batch_decode(predict, skip_special_tokens=False)

    prompt_id = np.where(predict_results.label_ids<0, 0, predict_results.label_ids)
    prompt = processing_class.batch_decode(prompt_id, skip_special_tokens=False)

    OUT = []
    for i in range(len(prompt)):
        if i%1 == 0:
            OUT.append({"label": prompt[i].replace('<|endoftext|>',"").replace('!',""), "predict": response[i].replace('<|endoftext|>',"").replace('!',"")})

    path = training_args.output_dir + "/result-2015.json"
    with open(path, 'w') as f:
        f.write(json.dumps(OUT, ensure_ascii=False, indent=4))



if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    main(script_args, training_args, model_args)
