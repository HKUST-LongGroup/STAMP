import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, LocalEntryNotFoundError
from transformers import AutoProcessor, DynamicCache
import numpy as np
import torch.nn.functional as F
from model.qwen25_changes import get_rope_index, SegQwenVL
import os  # 导入 os 模块
import json  # 导入 json 模块


def find_image_patch_info(image_pad_id, input_ids: torch.Tensor):
    """
    从后向前查找输入张量中连续的 image_pad_id，并返回其数量。

    参数:
        image_pad_id (int): 图像填充标记的 ID。
        input_ids (torch.Tensor): 输入的 ID 张量。

    返回:
        int: 连续图像补丁的数量。

    异常:
        RuntimeError: 如果在 input_ids 中找不到图像补丁 (<|image_pad|>)。
    """
    input_ids_list = input_ids.squeeze().tolist()

    # 将列表反转以便从后向前查找
    reversed_input_ids_list = input_ids_list[::-1]

    try:
        # 查找反转后第一个 image_pad_id 的位置
        start_idx_rev = reversed_input_ids_list.index(image_pad_id)
        end_idx_rev = start_idx_rev

        # 继续查找连续的 image_pad_id
        while end_idx_rev + 1 < len(reversed_input_ids_list) and reversed_input_ids_list[
            end_idx_rev + 1] == image_pad_id:
            end_idx_rev += 1

        num_patches = (end_idx_rev - start_idx_rev) + 1
        return num_patches
    except ValueError:
        raise RuntimeError("在 input_ids 中找不到图像补丁 (<|image_pad|>)。")


class GenerativeSegmenter:
    def __init__(self, model_path: str, min_pixels, max_pixels, **kwargs):
        min_pixels = min_pixels
        max_pixels = max_pixels
        self.device = kwargs.get("device_map", "cuda" if torch.cuda.is_available() else "cpu")

        # --- 新增的智能加载逻辑 ---
        adapter_config_path_local = os.path.join(model_path, "adapter_config.json")
        is_adapter = False
        adapter_config = None
        adapter_model_path = model_path  # 适配器路径就是传入的 model_path
        base_model_path = None
        if os.path.exists(adapter_config_path_local):
            print(f"检测到本地 PEFT 适配器配置: {adapter_config_path_local}.")
            is_adapter = True
            with open(adapter_config_path_local, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path")
        else:
            # Case 2: model_path 可能是一个 Hub ID (完整模型或适配器)
            # 尝试从 Hub 下载 adapter_config.json 来检查它是否存在
            try:
                adapter_config_file = hf_hub_download(
                    repo_id=model_path,
                    filename="adapter_config.json",
                    # token=kwargs.get("token") # 如果您的仓库是私有的，可能需要 token
                )
                print(f"检测到 Hugging Face Hub 上的 PEFT 适配器配置: {model_path}.")
                is_adapter = True
                with open(adapter_config_file, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)

            except (EntryNotFoundError, RepositoryNotFoundError, LocalEntryNotFoundError):
                # Case 3: 在 Hub 上找不到 adapter_config.json，或者 model_path 根本不是一个有效的 repo
                # (LocalEntryNotFoundError 捕获当 model_path 是一个本地路径但 *不是* 适配器的情况)
                print(f"未在 '{model_path}' 中检测到适配器配置。将尝试作为完整模型加载。")
                is_adapter = False
            except Exception as e:
                # 其他潜在问题 (例如网络错误)
                print(f"检查 '{model_path}' 适配器时出错: {e}. 将尝试作为完整模型加载。")
                is_adapter = False
        if is_adapter:
            # --- 修正后的适配器 (PEFT) 加载逻辑 ---

            # 1. 确定基础模型路径
            if not base_model_path:
                base_model_path = "Qwen/Qwen2-VL-7B-Instruct"  # 您的默认值
                print(f"警告: 无法从适配器配置中找到 'base_model_name_or_path'。将使用默认基础模型: '{base_model_path}'")

            # 2. 加载基础模型
            print(f"正在从 '{base_model_path}' 加载基础模型...")
            self.model = base_model = SegQwenVL.from_pretrained(
                base_model_path, torch_dtype="auto", trust_remote_code=True, **kwargs
            )

            # 3. 加载 PEFT 配置 (!!! 关键修复 !!!)
            # 我们不直接使用 adapter_config_file，而是使用 PeftConfig.from_pretrained
            # 这样它能正确加载，但我们稍后会覆盖缺失的字段
            print(f"正在从 '{adapter_model_path}' 加载 PEFT 配置...")
            peft_config = PeftConfig.from_pretrained(adapter_model_path)

            # 4. 手动注入缺失的训练参数
            # 这是您提供的确切列表
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            modules_to_save = ["embed_tokens", "lm_head", "classifier"]

            if not getattr(peft_config, "target_modules", None):
                print(f"警告: PEFT 配置缺少 'target_modules'。将手动设置为: {target_modules}")
                peft_config.target_modules = target_modules

            if not getattr(peft_config, "modules_to_save", None):
                print(f"警告: PEFT 配置缺少 'modules_to_save'。将手动设置为: {modules_to_save}")
                peft_config.modules_to_save = modules_to_save

            # 确保其他关键参数也存在 (可选但推荐)
            peft_config.r = getattr(peft_config, "r", 64)
            peft_config.lora_alpha = getattr(peft_config, "lora_alpha", 128)
            peft_config.task_type = getattr(peft_config, "task_type", "CAUSAL_LM")
            peft_config.bias = getattr(peft_config, "bias", "none")

            self.processor = AutoProcessor.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            self.tokenizer = self.processor.tokenizer
            self._add_special_tokens()
            # 5. 使用 PeftModel 附加适配器
            print("正在将适配器附加到基础模型...")
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_model_path,
                config=peft_config,  # 传入我们修正后的配置
            )


            print("适配器加载并附加完成。")

            # 6. 加载基础模型的 Processor

        else:
            print(f"未检测到 PEFT 适配器。将直接从 '{model_path}' 加载完整模型。")
            # 保持原始的直接加载方式
            self.model = SegQwenVL.from_pretrained(
                model_path, torch_dtype="auto", trust_remote_code=True, **kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels)
            self.tokenizer = self.processor.tokenizer
            self._add_special_tokens()
        # --- 智能加载逻辑结束 ---

        # TargetClass = type(self.model.model)
        # TargetClass.get_rope_index = get_rope_index


        # 获取关键token的ID
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("<|yes|>")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("<|no|>")
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids("<|seg|>")
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|mask|>")
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        self.eos_token_id = self.tokenizer.eos_token_id
        self.model.mask_token_id = self.mask_token_id
        if hasattr(self.model, "base_model"): # for peft model
            self.model.base_model.config.mask_token_id = self.mask_token_id

    def _add_special_tokens(self):
        special_tokens = {'additional_special_tokens': ["<|seg|>", "<|mask|>", "<|yes|>", "<|no|>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            print(f"添加了 {num_added} 个特殊 tokens。正在调整模型嵌入层大小...")
            self.model.resize_token_embeddings(len(self.tokenizer))
            # 检查调整后的大小是否与您的模型期望匹配
            print(
                f"调整后词汇表大小: {len(self.tokenizer)}, 模型嵌入层大小: {self.model.get_input_embeddings().weight.shape[0]}")
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def generate_with_segmentation(self, image: Image.Image, prompt: str):
        # ... (此方法保持不变) ...
        messages = [{"role": "user", "content": [{"image": image}, {"text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        merge_size = self.processor.image_processor.merge_size

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[1]
        image_grid_thw = inputs.get('image_grid_thw').to(self.device)  # Qwen2.5-VL may use this key
        attention_mask_raw = inputs['attention_mask'].to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            use_cache=True,
            return_dict_in_generate=True,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        sequence = outputs.sequences[0]
        full_past_key_values = outputs.past_key_values

        # 查找所有 <seg> token 的位置
        seg_indices = torch.where(sequence == self.seg_token_id)[0].tolist()

        all_segmentation_masks = []

        if not seg_indices:  # 如果没有分割任务
            generated_ids = sequence[prompt_len:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return None, response_text

        num_patches = find_image_patch_info(self.image_pad_id, inputs['input_ids'])

        # 遍历每个 <seg> token 并执行分割
        for i, idx in enumerate(seg_indices):
            sliced_len = idx + 1
            attention_mask = attention_mask_raw[:, :sliced_len]
            legacy_cache = full_past_key_values.to_legacy_cache()
            # 2. 对元组中的每个张量进行切片
            past_key_values_sliced = tuple(
                (
                    key_layer[:, :, :sliced_len, :],
                    value_layer[:, :, :sliced_len, :]
                )
                for key_layer, value_layer in legacy_cache
            )
            past_key_values_sliced = DynamicCache.from_legacy_cache(past_key_values_sliced)

            mask_query_ids = torch.full((1, num_patches), self.mask_token_id, dtype=torch.long, device=self.device)
            mask_query_attention_mask = torch.ones((1, num_patches + sliced_len - attention_mask[0].sum()),
                                                   dtype=torch.long, device=self.device)
            mask_query_attention_mask = torch.cat((attention_mask, mask_query_attention_mask), dim=1)
            mask_grid_thw = image_grid_thw[-1].clone()
            mask_grid_thw = mask_grid_thw.unsqueeze(0)

            mask_pre_ids = sequence.clone().unsqueeze(0)
            mask_ids = torch.cat([mask_pre_ids[0, :sliced_len], mask_query_ids[0]], dim=0).unsqueeze(0)
            seg_forward_outputs = self.model(
                input_ids=mask_ids,
                attention_mask=mask_query_attention_mask,
                image_grid_thw=image_grid_thw,
                pixel_values=inputs['pixel_values'],
                past_key_values=past_key_values_sliced,
                return_dict=True,
                do_classification=True
            )

            mask_logits = seg_forward_outputs.bi_logits[:, -num_patches:]

            segmentation_preds = (mask_logits > 0).long().squeeze().cpu()
            h_grid, w_grid = mask_grid_thw[0, 1:]
            h_grid, w_grid = int(h_grid / merge_size), int(w_grid / merge_size)
            segmentation_preds = segmentation_preds.view(h_grid, w_grid)
            all_segmentation_masks.append(segmentation_preds)

        generated_ids = sequence[prompt_len:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return all_segmentation_masks, response_text


# --- 主程序入口 (无需更改) ---
if __name__ == '__main__':
    # ... (保持不变) ...
    # segmenter = GenerativeSegmenter(
    #     "/efficient_sag4text/new_train/final_model/",
    #     device_map="cuda"
    # )

    segmenter = GenerativeSegmenter(
        "/efficient_sag4text/new_train_qwen_2b/checkpoint-6000/",
        device_map="cuda",
        min_pixels=1024 * 28 * 28,
        max_pixels=1280 * 28 * 28
    )

    img_url = "/apdcephfs_qy4/share_302593112/realzliu/dataset_open/lisa_dataset/refer_seg/images/mscoco/images/train2014/COCO_train2014_000000000064.jpg"
    try:
        image = Image.open(img_url).convert('RGB')
    except Exception:
        print("图片下载失败，使用一个虚拟图片代替。")
        image = Image.new('RGB', (448, 448), color='brown')

    # 案例1：单个分割
    print("=" * 50)
    print("案例 1: 单个分割")
    print("=" * 50)
    prompt_seg_single = "Could you provide the segmentation mask for 'truck' in this image? <|seg|> Could you provide the segmentation mask for 'truck' in this image? Could you provide the segmentation mask for 'truck' in this image?"
    segmentation_masks, response_text = segmenter.generate_with_segmentation(image, prompt_seg_single)

    if segmentation_masks:
        print(f"\n--- 分割完成 (共 {len(segmentation_masks)} 个) ---")
        for i, segmentation_mask in enumerate(segmentation_masks):
            segmentation_mask = F.interpolate(segmentation_mask.unsqueeze(0).unsqueeze(0).float(),
                                              size=image.size[::-1], mode='nearest').squeeze().numpy()
            import matplotlib.pyplot as plt

            fusion = np.array(image) / 255 * 0.7 + np.stack([segmentation_mask] * 3, axis=-1) * 0.3
            plt.imshow(fusion)
            plt.savefig(f"segmentation_single_{i}.png")
            plt.close()
    print(f"\n模型生成的后续文本:\n'{response_text}'")