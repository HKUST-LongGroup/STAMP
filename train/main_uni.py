import deepspeed
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.grefer_seg_dataset import grefcocoValDataset
from dataset.refer_seg_dataset import ValDataset
from model.segment_anything import sam_model_registry, SamPredictor
from train.val_callback import CustomDataset, collate_fn, InProcessEvaluationCallback

deepspeed.ops.op_builder.CPUAdamBuilder().load()
from typing import List, Dict, Any
import os

import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    TrainingArguments, TrainerCallback, TrainerState, TrainerControl
)
from peft import LoraConfig, get_peft_model
import json
from qwen_vl_utils import process_vision_info
from trl import SFTConfig
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

from .seg_trainer import SegmentationSFTTrainer
from model.qwen_changes import get_rope_index, SegQwenVL
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "-1") in ["-1", "0"]
# random.seed(42)

class CustomLogCallback(TrainerCallback):
    """
    A custom callback to log additional loss components.
    """
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # `kwargs` usually contains the 'logs' dictionary and 'model'
        # We need to get the cached metrics from the trainer instance
        if hasattr(self.trainer, "_custom_log_metrics") and self.trainer._custom_log_metrics:
            # Add our custom metrics to the logs dictionary about to be recorded
            logs = kwargs.get("logs", {})
            logs.update(self.trainer._custom_log_metrics)
            # Clear to be safe
            self.trainer._custom_log_metrics = {}

class CustomDataCollator(DataCollatorForVisionLanguageModeling):
    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Call the parent class method to get the batch input_ids, etc.
        # raw_examples = examples.copy()
        batch = super().torch_call(examples)

        # If no masks are provided, do nothing
        # if "masks" not in examples[0]:
        #     return batch

        # Adjust labels based on masks
        all_masks = []
        all_images = []
        for i, example in enumerate(examples):
            imgs = [Image.open(m) for m in example['images']]
            # to tensor
            all_images.append(imgs)
            # raw_example = raw_examples[i]
            # if 'masks' in raw_example:
            #     example['masks'] = raw_example['masks']
            if 'masks' in example and example['masks'] is not None:
                # Here we assume masks is a list of the same length as input_ids, containing 0s and 1s
                masks = [Image.open(m) for m in example['masks']]
                # to tensor
                all_masks.append(masks)
            else:
                all_masks.append([])

        batch["masks"] = all_masks
        batch["all_images"] = all_images

        return batch



class QwenVLSFTTrainer:
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        # --- 2. Load model and processor (using 4-bit quantization to save memory) ---
        if IS_MAIN_PROCESS:
            print("--- Loading model and processor ---")
        min_pixels = 1024 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels)

        self.model = SegQwenVL.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="cpu",
            trust_remote_code=True,
            quantization_config=None  # If you have enough VRAM, you can skip quantization; otherwise, configure BitsAndBytesConfig
        )
        TargetClass = type(self.model.model)
        TargetClass.get_rope_index = get_rope_index
        # ##################################################################
        # #                   >>>>> New code starts <<<<<                     #
        # ##################################################################

        if IS_MAIN_PROCESS:
            print("--- Adding special tokens for segmentation task ---")
        # Define the special tokens we need for the segmentation task
        special_tokens = {'additional_special_tokens': ["<|seg|>", "<|mask|>", "<|yes|>", "<|no|>"]}

        # Add the special tokens to the tokenizer
        num_added_tokens = self.processor.tokenizer.add_special_tokens(special_tokens)

        if num_added_tokens > 0:
            if IS_MAIN_PROCESS:
                print(f"--- Added {num_added_tokens} new special tokens to the tokenizer. ---")
            # If new tokens were successfully added, the model's embedding size must be adjusted
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
            if IS_MAIN_PROCESS:
                print("--- Resized model token embeddings to match the new tokenizer size. ---")

        # ##################################################################
        # #                   >>>>>  New code ends <<<<<                     #
        # ##################################################################

        # Ensure the tokenizer has a pad_token, which is especially important after adding new tokens
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
            if IS_MAIN_PROCESS:
                print(f"--- Set tokenizer pad_token_id to eos_token_id: {self.processor.tokenizer.eos_token_id} ---")

        if IS_MAIN_PROCESS:
            print("--- Model and processor loaded and configured successfully ---")
        data_collator = CustomDataCollator(processor=self.processor)
        self.collator = data_collator
        self.model.mask_token_id = mask_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|mask|>")

    def _create_dummy_dataset(self):
        """
        Create a combined and shuffled dataset from multiple JSON files.
        In a real application, you would load data from your data sources.
        The dataset format requires a Hugging Face Dataset object containing "image" and "text" columns.
        The "text" column should be a complete dialogue string following the model's chat template.
        """
        if IS_MAIN_PROCESS:
            print("--- Creating a combined and shuffled dataset from multiple JSON files ---")

        base_path = 'playground/data/json_files/'
        json_files = [
            'all_valid_llava_data_1000.json',
            'refclef_formatted_all_sentences_doubled_mp.json',
            'refcocog_formatted_all_sentences_doubled_mp.json',
            'refcoco_formatted_all_sentences_doubled_mp.json',
            'refcoco+_formatted_all_sentences_doubled_mp.json'
        ]

        # Construct full file paths
        file_paths = [os.path.join(base_path, f) for f in json_files]
        # Load all data
        all_data = []
        for file_path in file_paths:
            if IS_MAIN_PROCESS:
                print(f"--- Loading {file_path} ---")
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'llava' not in file_path:
                    data = data * 3
                all_data.extend(data)

        if IS_MAIN_PROCESS:
            print(f"--- Total records loaded: {len(all_data)} ---")

        # Shuffle data
        # if IS_MAIN_PROCESS:
        #     print("--- Shuffling the dataset ---")
        # random.shuffle(all_data)

        processed_data = []
        IMAGE_RAW_ROOT_PATH = '/apdcephfs_qy4/share_302593112/realzliu/dataset_open/lisa_dataset'
        IMAGE_ROOT_PATH = '/efficient_sag4text/seg_data'
        for example in tqdm(all_data):
            # 1. Process image paths
            images = example['images']
            images = [os.path.join(IMAGE_RAW_ROOT_PATH, i) for i in images]
            images = [i.replace(IMAGE_RAW_ROOT_PATH, IMAGE_ROOT_PATH) for i in images]
            images = [i.replace('/coco_2014/', '/mscoco/images/') for i in images]
            example['images'] = images

            # 2. Normalize the 'masks' key
            # This is the key to solving the problem: ensure each example has a 'masks' key
            if 'masks' not in example:
                example['masks'] = None

            processed_data.append(example)


        # Create Dataset object
        ds = Dataset.from_list(all_data)
        # Apply formatting function
        # formatted_ds = ds.map(format_chat_template)

        if IS_MAIN_PROCESS:
            print(f"--- Dataset created. Sample formatted text:\n{ds[0]['messages']} ---")

        return ds

    def train(self):
        """
        Configure and start the training process.
        """
        # --- 3. Prepare dataset ---
        train_dataset = self._create_dummy_dataset()

        # --- 4. Configure PEFT (LoRA) ---
        if IS_MAIN_PROCESS:
            print("--- Configuring PEFT (LoRA) ---")
        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_rslora=True,
            modules_to_save=["embed_tokens", "lm_head", "classifier"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # --- 5. Configure training parameters ---
        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=2,  
            per_device_train_batch_size=8,  
            gradient_accumulation_steps=4,  
            learning_rate=3e-5, 
            lr_scheduler_type="linear", 
            warmup_ratio=0.03,
            weight_decay=0.0,
            max_grad_norm=1.0,
            logging_steps=1,  
            save_steps=1000,  
            bf16=True,  
            tf32=True,  
            remove_unused_columns=False,  
            report_to="wandb",  
            max_length=4096,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
        )

        # --- 6. Initialize SFTTrainer ---
        if IS_MAIN_PROCESS:
            print("--- Initializing SFTTrainer ---")
        trainer = SegmentationSFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.processor,
            data_collator=self.collator,
        )

        if trainer.is_world_process_zero():
            print("--- Starting training ---")
        trainer.train()
        if trainer.is_world_process_zero():
            print("--- Training finished ---")

        final_model_path = f"{self.output_dir}/final_model"
        trainer.save_model(final_model_path)
        if trainer.is_world_process_zero():
            print(f"--- Model saved to {final_model_path} ---")


if __name__ == '__main__':
    model_name = 'Qwen/Qwen2-VL-2B-Instruct'  # or your local model path
    trainer = QwenVLSFTTrainer(
        model_name=model_name,
        output_dir="output/qwen_vl_seg_sft/uni/",
    )
    trainer.train()