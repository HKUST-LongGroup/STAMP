import torch
from PIL import Image
from transformers import AutoProcessor, DynamicCache
import numpy as np
import torch.nn.functional as F
from model.qwen_changes import get_rope_index, SegQwenVL
import os  
import json  
import time  


def find_image_patch_info(image_pad_id, input_ids: torch.Tensor):
    """
    From the end to the beginning, find consecutive image_pad_id in the input tensor and return their count.

    Parameters:
        image_pad_id (int): The ID of the image padding token.
        input_ids (torch.Tensor): The input tensor of IDs.

    Returns:
        int: The number of consecutive image patches.

    Raises:
        RuntimeError: If no image patches (<|image_pad|>) are found in input_ids.
    """
    input_ids_list = input_ids.squeeze().tolist()

    # Reverse the list to search from the end to the beginning
    reversed_input_ids_list = input_ids_list[::-1]

    try:
        # Find the first occurrence of image_pad_id in the reversed list
        start_idx_rev = reversed_input_ids_list.index(image_pad_id)
        end_idx_rev = start_idx_rev

        # Continue to find consecutive image_pad_id
        while end_idx_rev + 1 < len(reversed_input_ids_list) and reversed_input_ids_list[
            end_idx_rev + 1] == image_pad_id:
            end_idx_rev += 1

        num_patches = (end_idx_rev - start_idx_rev) + 1
        return num_patches
    except ValueError:
        raise RuntimeError("No image patches (<|image_pad|>) found in input_ids.")


class GenerativeSegmenter:
    def __init__(self, model_path: str, min_pixels, max_pixels, **kwargs):
        min_pixels = min_pixels
        max_pixels = max_pixels
        self.device = kwargs.get("device_map", "cuda" if torch.cuda.is_available() else "cpu")

        # --- New intelligent loading logic ---
        adapter_config_path = os.path.join(model_path, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            print(f"Detected PEFT adapter configuration: {adapter_config_path}. Will load base model first, then load adapter.")
            # Read the base model path from the adapter configuration
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
            # Base model path, if not present in the config, you need to specify it manually
            base_model_path = adapter_config.get("base_model_name_or_path")
            if not base_model_path:
                # ********************************************************************************
                # ** Important: If adapter_config.json does not contain base_model_name_or_path,
                # ** please manually specify the correct base model name or path here
                # ** Based on your previous error messages, the base model is likely "Qwen/Qwen2-VL-7B-Instruct"
                # ********************************************************************************
                base_model_path = "Qwen/Qwen2-VL-7B-Instruct"
                print(f"Warning: 'base_model_name_or_path' not found in adapter configuration. Using default base model: '{base_model_path}'")
            # 1. Load the base model
            print(f"Loading base model from '{base_model_path}'...")
            self.model = SegQwenVL.from_pretrained(
                base_model_path, 
                torch_dtype="auto", 
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",  
                **kwargs
            )
            self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True,
                                                           min_pixels=min_pixels, max_pixels=max_pixels)
            self.tokenizer = self.processor.tokenizer
            self._add_special_tokens()
            # 2. Load the adapter
            print(f"Loading adapter from '{model_path}'...")
            self.model.load_adapter(model_path)

        else:
            print(f"No PEFT adapter detected. Loading full model directly from '{model_path}'.")
            # Keep the original direct loading method
            self.model = SegQwenVL.from_pretrained(
                model_path, 
                torch_dtype="auto", 
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",  
                **kwargs
            )
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, min_pixels=min_pixels,
                                                           max_pixels=max_pixels)
            self.tokenizer = self.processor.tokenizer
            self._add_special_tokens()
        # --- Intelligent loading logic ends ---

        TargetClass = type(self.model.model)
        TargetClass.get_rope_index = get_rope_index

        # Get key token IDs
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("<|yes|>")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("<|no|>")
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids("<|seg|>")
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|mask|>")
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        self.eos_token_id = self.tokenizer.eos_token_id
        self.model.mask_token_id = self.mask_token_id

    def _add_special_tokens(self):
        special_tokens = {'additional_special_tokens': ["<|seg|>", "<|mask|>", "<|yes|>", "<|no|>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            print(f"Added {num_added} special tokens. Resizing model embedding layer...")
            self.model.resize_token_embeddings(len(self.tokenizer))
            # Check if the resized size matches your model's expectations
            print(
                f"Resized vocabulary size: {len(self.tokenizer)}, Model embedding layer size: {self.model.get_input_embeddings().weight.shape[0]}")
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def generate_with_segmentation(self, image: Image.Image, prompt: str):
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

        # Find all <seg> token positions
        seg_indices = torch.where(sequence == self.seg_token_id)[0].tolist()

        all_segmentation_masks = []
        seg_forward_times = []  # Initialize list to store times
        if not seg_indices:  # If there are no segmentation tasks
            generated_ids = sequence[prompt_len:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return None, response_text

        num_patches = find_image_patch_info(self.image_pad_id, inputs['input_ids'])

        # Iterate over each <seg> token and perform segmentation
        for i, idx in enumerate(seg_indices):
            sliced_len = idx + 1
            attention_mask = attention_mask_raw[:, :sliced_len]
            legacy_cache = full_past_key_values.to_legacy_cache()
            # 2. Slice each tensor in the tuple
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

