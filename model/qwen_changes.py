from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import torch
from torch import nn
from transformers import DynamicCache
from .modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from transformers.masking_utils import create_causal_mask
from transformers.utils import ModelOutput


def replace_token_pair_vectorized(
    input_ids: torch.Tensor,
    seg_start_token_id: int,
    seg_holder_token_id: int,
    vision_start_token_id: int,
    image_token_id: int,
) -> torch.Tensor:
    modified_ids = input_ids.clone()

    #creating aligned views of current and next tokens
    current_tokens = modified_ids[..., :-1]
    next_tokens = modified_ids[..., 1:]

    # parallel find all positions where (current == start) & (next == holder)
    mask = (current_tokens == seg_start_token_id) & (next_tokens == seg_holder_token_id)

    #   Use the mask to perform all replacements at once, in parallel
    modified_ids[..., :-1][mask] = vision_start_token_id
    modified_ids[seg_holder_token_id == modified_ids] = image_token_id

    return modified_ids, mask.sum()

import torch

def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        seg_start_token_id: Optional[int] = None,
        seg_holder_token_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        input_ids = input_ids.clone()
        if seg_start_token_id is not None and seg_holder_token_id is not None:
            input_ids, num = replace_token_pair_vectorized(input_ids, seg_start_token_id, seg_holder_token_id,
                                                           vision_start_token_id, image_token_id)
            mask_grid_thw = image_grid_thw[-1].clone()
            mask_grid_thw = mask_grid_thw.unsqueeze(0).repeat([num, 1])
            image_grid_thw = torch.cat((image_grid_thw, mask_grid_thw), dim=0)

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            if isinstance(attention_mask, dict):
                attention_mask = attention_mask['raw_attention']
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i].to(input_ids.device) == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

def get_rope_index_2_5(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        seg_start_token_id: Optional[int] = None,
        seg_holder_token_id: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id
    input_ids = input_ids.clone()
    if seg_start_token_id is not None and seg_holder_token_id is not None:
        input_ids, num = replace_token_pair_vectorized(input_ids, seg_start_token_id, seg_holder_token_id,
                                                  vision_start_token_id, image_token_id)
        mask_grid_thw = image_grid_thw[-1].clone()
        mask_grid_thw = mask_grid_thw.unsqueeze(0).repeat([num, 1])
        image_grid_thw = torch.cat((image_grid_thw, mask_grid_thw), dim=0)

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                ## normalize type, send to device.
                second_per_grid_t = torch.as_tensor(
                    second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                )

                time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas

@dataclass
class CustomModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    bi_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


import torch


def create_bidirectional_lookup_function(seg_mask_tensor: torch.Tensor) -> Callable:

    def lookup_function(batch_idx, head_idx, q_idx, kv_idx) -> bool:
        is_query_in_seg = seg_mask_tensor[batch_idx, q_idx]

        return is_query_in_seg

    return lookup_function

def _create_hybrid_mask_and_dependencies(
        self,
        seg_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
):


    bidirectional_mask_fn = create_bidirectional_lookup_function(seg_mask)

    use_cache = kwargs.get('use_cache', None)
    if self.is_gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    past_key_values = kwargs.get('past_key_values', None)
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    cache_position = kwargs.get('cache_position', None)
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        local_position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        local_position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    else:
        local_position_ids = position_ids

    if local_position_ids.ndim == 3 and local_position_ids.shape[0] == 4:
        text_position_ids = local_position_ids[0]
        final_position_ids = local_position_ids[1:]
    else:
        text_position_ids = local_position_ids[0]
        final_position_ids = position_ids  

    mask_kwargs = {
        "config": self.config,
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": text_position_ids,
        "or_mask_function": bidirectional_mask_fn,
    }
    hybrid_attention_mask = create_causal_mask(**mask_kwargs)

    return hybrid_attention_mask, final_position_ids, past_key_values, use_cache, cache_position

class SegQwenVL(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.model._create_hybrid_mask_and_dependencies = _create_hybrid_mask_and_dependencies.__get__(self)
        self.model.get_rope_index = get_rope_index.__get__(self)

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.FloatTensor = None, pixel_values: torch.FloatTensor = None,
                position_ids=None, labels: torch.LongTensor = None, do_classification: bool=False, output_hidden_states=False, **kwargs,):

        if do_classification:  
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            image_embeds = self.model.get_image_features(pixel_values, kwargs['image_grid_thw'])
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            seg_mask = (input_ids == self.mask_token_id)

            inputs_embeds[seg_mask] = inputs_embeds[seg_mask] + image_embeds[-seg_mask.sum():]

            outputs = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=None,
                output_hidden_states=True,
                position_ids=position_ids,
                seg_mask=seg_mask,
                **kwargs,
            )
            last_hidden_state = outputs.hidden_states[-1]
            logits = self.classifier(last_hidden_state)

            return CustomModelOutput(
                bi_logits=logits,
                # hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        else:
            if labels is not None:
                output_hidden_states = True

            original_output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=output_hidden_states,
                position_ids=position_ids,
                **kwargs,
            )
            if labels is not None:
                last_hidden_state = original_output.hidden_states[-1]
                dummy_logits = self.classifier(last_hidden_state) 
                if hasattr(original_output, 'loss') and original_output.loss is not None:
                    dummy_loss = dummy_logits[0, 0].sum() * 0.0
                    original_output.loss += dummy_loss

            return original_output

