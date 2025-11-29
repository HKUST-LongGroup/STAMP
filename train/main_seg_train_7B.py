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
# import random
# --- Helper to check for main process ---
# 在 Trainer 初始化前，通过环境变量判断是否为主进程
# 在分布式训练启动器（如 torchrun）中，主进程的 LOCAL_RANK 通常是 '0'
# 在非分布式场景下，此环境变量不存在，默认为 '-1'
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "-1") in ["-1", "0"]
# random.seed(42)

class CustomLogCallback(TrainerCallback):
    """
    一个自定义的回调，用于在日志中记录额外的损失分量。
    """
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # `kwargs` 中通常包含 'logs' 字典和 'model'
        # 我们需要从 trainer 实例中获取暂存的指标
        if hasattr(self.trainer, "_custom_log_metrics") and self.trainer._custom_log_metrics:
            # 将我们的自定义指标添加到即将被记录的 logs 字典中
            logs = kwargs.get("logs", {})
            logs.update(self.trainer._custom_log_metrics)
            # 清除，以防万一
            self.trainer._custom_log_metrics = {}

class CustomDataCollator(DataCollatorForVisionLanguageModeling):
    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 调用父类的方法，获取批处理后的input_ids等
        # raw_examples = examples.copy()
        batch = super().torch_call(examples)

        # 如果没有提供mask，则不进行任何操作
        # if "masks" not in examples[0]:
        #     return batch

        # 根据mask来调整labels
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
                # 这里假设 masks 是一个与 input_ids 长度相同的列表，包含0和1
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
        # --- 2. 加载模型和处理器 (使用4-bit量化以节省显存) ---
        if IS_MAIN_PROCESS:
            print("--- Loading model and processor ---")
        min_pixels = 1024 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels)

        self.model = SegQwenVL.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="cpu",
            trust_remote_code=True,
            quantization_config=None  # 如果显存充足，可以不用量化；若要用，请配置BitsAndBytesConfig
        )
        TargetClass = type(self.model.model)
        TargetClass.get_rope_index = get_rope_index
        # ##################################################################
        # #                   >>>>> 新增代码开始 <<<<<                     #
        # ##################################################################

        if IS_MAIN_PROCESS:
            print("--- Adding special tokens for segmentation task ---")
        # 定义我们需要用于分割任务的特殊token
        special_tokens = {'additional_special_tokens': ["<|seg|>", "<|mask|>", "<|yes|>", "<|no|>"]}

        # 将特殊token添加到tokenizer中
        num_added_tokens = self.processor.tokenizer.add_special_tokens(special_tokens)

        if num_added_tokens > 0:
            if IS_MAIN_PROCESS:
                print(f"--- Added {num_added_tokens} new special tokens to the tokenizer. ---")
            # 如果成功添加了新的token，必须调整模型嵌入层的大小
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
            if IS_MAIN_PROCESS:
                print("--- Resized model token embeddings to match the new tokenizer size. ---")

        # ##################################################################
        # #                   >>>>>  新增代码结束 <<<<<                     #
        # ##################################################################

        # 确保tokenizer有pad_token，这在添加新token后尤为重要
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
        创建一个从多个JSON文件加载并打乱的数据集。
        在实际应用中，你需要从你的数据源加载数据。
        数据集格式要求：一个包含 "image" 和 "text" 列的 Hugging Face Dataset 对象。
        "text" 列应为遵循模型聊天模板的完整对话字符串。
        """
        if IS_MAIN_PROCESS:
            print("--- Creating a combined and shuffled dataset from multiple JSON files ---")

        base_path = '/apdcephfs_nj4/share_300377003/realzliu/data/json_files/'
        json_files = [
            # 'llava_v1_5_mix665k_transfered.json',
            'refclef_formatted_all_sentences_doubled_mp.json',
            'refcocog_formatted_all_sentences_doubled_mp.json',
            'refcoco_formatted_all_sentences_doubled_mp.json',
            'refcoco+_formatted_all_sentences_doubled_mp.json'
        ]

        # 构建完整的文件路径
        file_paths = [os.path.join(base_path, f) for f in json_files]
        # file_paths = ['/efficient_sag4text/playground/data/filtered_data_under_1024.json']
        # 加载所有数据
        all_data = []
        for file_path in file_paths:
            if IS_MAIN_PROCESS:
                print(f"--- Loading {file_path} ---")
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)

        if IS_MAIN_PROCESS:
            print(f"--- Total records loaded: {len(all_data)} ---")

        # 打乱数据
        # if IS_MAIN_PROCESS:
        #     print("--- Shuffling the dataset ---")
        # random.shuffle(all_data)

        processed_data = []
        IMAGE_RAW_ROOT_PATH = '/apdcephfs_qy4/share_302593112/realzliu/dataset_open/lisa_dataset'
        # IMAGE_ROOT_PATH = '/efficient_sag4text/seg_data'
        IMAGE_ROOT_PATH = IMAGE_RAW_ROOT_PATH
        for example in tqdm(all_data):
            # 1. 处理图像路径
            images = example['images']
            images = [os.path.join(IMAGE_RAW_ROOT_PATH, i) for i in images]
            images = [i.replace(IMAGE_RAW_ROOT_PATH, IMAGE_ROOT_PATH) for i in images]
            images = [i.replace('/coco_2014/', '/mscoco/images/') for i in images]
            example['images'] = images

            # 2. 规范化 'masks' 键
            # 这是解决问题的关键：确保每个 example 都有 'masks' 键
            if 'masks' not in example:
                example['masks'] = None

            processed_data.append(example)

        # 将对话数据转换为模型需要的格式化文本
        # def format_chat_template(example):
        #     # 将 'conversations' 列表转换为 apply_chat_template 需要的格式
        #     # messages = example["messages"]
        #     images = example['images']
        #     # 注意：这里的路径替换逻辑需要根据您的实际文件存储结构来确定是否正确
        #     images = [os.path.join('/apdcephfs_qy4/share_302593112/realzliu/dataset_open/lisa_dataset', i) for i in
        #               images]
        #     images = [i.replace('/coco_2014/', '/mscoco/images/') for i in images]
        #     example['images'] = images
        #     if 'masks' not in example:
        #         example['masks'] = None
        #     return example

        # 创建Dataset对象
        ds = Dataset.from_list(all_data)
        # 应用格式化函数
        # formatted_ds = ds.map(format_chat_template)

        if IS_MAIN_PROCESS:
            print(f"--- Dataset created. Sample formatted text:\n{ds[0]['messages']} ---")

        return ds

    def train(self):
        """
        配置并启动训练过程。
        """
        # --- 3. 准备数据集 ---
        train_dataset = self._create_dummy_dataset()
        ############ for eval
        # SPLIT_OPTIONS = [
        #     "refcoco|unc|val", "refcoco|unc|testA", "refcoco|unc|testB",
        #     "refcoco+|unc|val", "refcoco+|unc|testA", "refcoco+|unc|testB",
        #     "refcocog|umd|val", "refcocog|umd|test"
        # ]
        # accelerator = Accelerator()
        # eval_dataloaders = {}
        # if accelerator.is_main_process:
        #     print("Preparing evaluation dataloaders...")
        #
        # sam_path = None
        # image_folder = None
        # for split in SPLIT_OPTIONS:
        #     if "grefcoco" in split:
        #         val_dataset = grefcocoValDataset(image_folder, split)
        #     else:
        #         val_dataset = ValDataset(image_folder, split)
        #
        #     custom_eval_dataset = CustomDataset(val_dataset)
        #     eval_loader = DataLoader(
        #         custom_eval_dataset, batch_size=1, collate_fn=collate_fn, num_workers=4
        #     )
        #     # 关键: SFTTrainer 内部的 accelerator 会自动处理数据加载器
        #     # 所以这里我们暂时不需要手动 prepare()，回调会使用 trainer 自身的 accelerator
        #     eval_dataloaders[split] = eval_loader
        #
        # sam_predictor = None
        # if sam_path is not None:
        #     sam = sam_model_registry["vit_h"](checkpoint=sam_path)
        #     # SAM 模型不需要训练，所以直接移动到 accelerator 管理的设备
        #     sam = sam.to(accelerator.device)
        #     sam_predictor = SamPredictor(sam)
        #
        # evaluation_callback = InProcessEvaluationCallback(
        #     eval_dataloaders=eval_dataloaders,
        #     sam_predictor=sam_predictor
        # )
        #
        # if accelerator.is_main_process:
        #     print("Evaluation dataloaders prepared.")
        #
        # ############ for eval end
        # --- 4. 配置PEFT (LoRA) ---
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

        # --- 5. 配置训练参数 ---
        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=5,  # 训练轮数
            per_device_train_batch_size=8,  # 每个设备的批处理大小
            gradient_accumulation_steps=4,  # 梯度累积步数
            learning_rate=3e-5,  # 学习率
            lr_scheduler_type="linear",  # 学习率调度器
            warmup_ratio=0.03,
            weight_decay=0.0,
            max_grad_norm=1.0,
            logging_steps=1,  # 每隔多少步记录一次日志
            save_steps=1000,  # 每隔多少步保存一次模型
            bf16=True,  # 如果GPU支持，使用bfloat16
            tf32=True,  # 如果GPU支持，使用tf32
            remove_unused_columns=False,  # 需要保留 image 列
            report_to="wandb",  # 不上报到wandb等,
            max_length=None,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
        )

        # --- 6. 初始化 SFTTrainer ---
        if IS_MAIN_PROCESS:
            print("--- Initializing SFTTrainer ---")
        trainer = SegmentationSFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.processor,
            data_collator=self.collator,
            # callbacks=[evaluation_callback]
            # callbacks=[CustomLogCallback],
            peft_config=peft_config,  # 兼容旧版TRL
            # max_seq_length=10240, # 兼容旧版TRL
        )
        # custom_callback_instance = CustomLogCallback(trainer)
        # trainer.add_callback(custom_callback_instance)

        # --- 7. 开始训练 ---
        # 使用 trainer.is_world_process_zero() 是最推荐的方式
        if trainer.is_world_process_zero():
            print("--- Starting training ---")
        trainer.train()
        if trainer.is_world_process_zero():
            print("--- Training finished ---")

        # --- 8. 保存最终模型 ---
        final_model_path = f"{self.output_dir}/final_model"
        trainer.save_model(final_model_path)
        if trainer.is_world_process_zero():
            print(f"--- Model saved to {final_model_path} ---")


if __name__ == '__main__':
    # model_name = "/efficient_sag4text/new_train/final_model/"
    model_name = 'Qwen/Qwen2-VL-7B-Instruct'  # or your local model path
    trainer = QwenVLSFTTrainer(
        model_name=model_name,
        output_dir="/efficient_sag4text/new_train_qwen_7b/",
    )
    trainer.train()