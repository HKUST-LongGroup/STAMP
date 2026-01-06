import types
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import torch
from torch import nn
from transformers import DynamicCache
try:
    from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, logger
except:
    from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.masking_utils import create_causal_mask
from transformers.utils import ModelOutput


def replace_token_pair_vectorized(
        input_ids: torch.Tensor,
        seg_start_token_id: int,
        seg_holder_token_id: int,
        vision_start_token_id: int,
        image_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    功能：
    1. 找到 [Start, Holder...] 并替换为 VisionStart 和 ImageToken。
    2. 返回全局视角的 insertion_indices，用于在展平的 image_grid_thw 中索引。
    """
    modified_ids = input_ids.clone()

    # 1. 安全检查：(current == Start) & (next == Holder)
    current_tokens = modified_ids[..., :-1]
    next_tokens = modified_ids[..., 1:]
    start_mask = (current_tokens == seg_start_token_id) & (next_tokens == seg_holder_token_id)

    # 2. 计算全局 Vision Index (Global Cumulative Sum)

    # 2.1 基础：计算行内的 Vision 数量 (B, L)
    is_vision = (input_ids == vision_start_token_id).long()
    local_vision_counts = is_vision.cumsum(dim=-1)

    # 2.2 偏移：计算每一行之前所有行包含的 Vision 总数
    # (B,) -> [Count_Row0, Count_Row1, ...]
    row_totals = is_vision.sum(dim=-1)

    # 2.3 构造偏移量向量 (B,)
    # shift right: [0, Row0, Row0+Row1, ...]
    batch_offsets = torch.zeros_like(row_totals)
    batch_offsets[1:] = row_totals[:-1].cumsum(dim=0)

    # 2.4 计算全局计数 (B, L)
    # 广播相加：(B, L) + (B, 1)
    global_vision_counts = local_vision_counts + batch_offsets.unsqueeze(1)

    # 3. 获取 insertion_indices
    # start_mask 的形状是 (B, L-1)，对应取 global_vision_counts 的前 L-1 列
    # [mask] 操作会自动将结果展平为 1D Tensor，包含所有 batch 中被激活位置的值
    insertion_indices = global_vision_counts[..., :-1][start_mask]

    # 4. 执行替换
    # 4.1 替换 Start
    modified_ids[..., :-1][start_mask] = vision_start_token_id
    # 4.2 替换 Holder (全局)
    modified_ids[modified_ids == seg_holder_token_id] = image_token_id

    return modified_ids, start_mask.sum(), insertion_indices

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
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
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
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id
    raw = input_ids
    input_ids = input_ids.clone()
    if seg_start_token_id is not None and seg_holder_token_id is not None:
        input_ids, num, idx = replace_token_pair_vectorized(input_ids, seg_start_token_id, seg_holder_token_id,
                                                  vision_start_token_id, image_token_id)
        if num != 0:
            mask_grids = image_grid_thw[idx-1].clone()
            num_images = image_grid_thw.shape[0]
            image_keys = torch.arange(num_images, device=image_grid_thw.device) * 2

            # Mask 的键：对应的 Image 键 + 1 (即 1, 3, 5...)
            # idx=1 代表第1张图(key=0)，Mask key = (1-1)*2 + 1 = 1 -> 排在 0 后面
            mask_keys = (idx - 1) * 2 + 1

            # 3. 拼接并排序
            all_grids = torch.cat([image_grid_thw, mask_grids], dim=0)
            all_keys = torch.cat([image_keys, mask_keys], dim=0)

            # argsort 得到正确的顺序索引
            sort_indices = torch.argsort(all_keys)

            # 4. 得到最终的 grid list
            image_grid_thw = all_grids[sort_indices]
    # print((input_ids != raw).sum())
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
    """
    Returns a function that performs an index-based lookup to decide
    if a query-key pair should have bidirectional attention.
    """

    # This is the actual function that will be passed as `or_mask_function`.
    # It "closes over" and remembers the `seg_mask_tensor`.
    def lookup_function(batch_idx, head_idx, q_idx, kv_idx) -> bool:
        """
        正确的混合注意力逻辑：
        只有当“观察者”（Query Token）本身位于 seg_mask 区域时，
        我们才赋予它“看到一切”的特权。
        如果“观察者”是普通文本，它不获得任何特权（返回False），
        因此它的注意力行为将完全由外部的标准因果掩码来决定。
        """
        # 检查“观察者”（query token）是否在特殊区域内
        is_query_in_seg = seg_mask_tensor[batch_idx, q_idx]

        # 如果是，就授予额外权限（允许这次查询）；如果不是，就不授予。
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
    """
    根据 seg_mask 创建混合注意力掩码，并准备所有相关的依赖项。
    此方法封装了所有临时变量的创建和修改，以避免副作用。
    """
    # --- 这里的逻辑与你提供的CHANGE块完全相同 ---

    bidirectional_mask_fn = create_bidirectional_lookup_function(seg_mask)

    use_cache = kwargs.get('use_cache', None)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    past_key_values = kwargs.get('past_key_values', None)
    cache_position = kwargs.get('cache_position', None)


    if self.is_gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
    # where each dim indicates visual spatial positions for temporal/height/width grids.
    # There are two scenarios when FA2-like packed masking might be activated.
    # 1. User specifically passed packed `position_ids` and no attention mask.
    #    In this case we expect the useer to create correct position ids for all 3 grids
    #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
    # 2. User runs forward with no attention mask and no position ids. In this case, position ids
    #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
    #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
    #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    mask_kwargs = {
        "config": self.config,
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": text_position_ids,
        "or_mask_function": bidirectional_mask_fn,
    }
    # print(mask_kwargs)
    hybrid_attention_mask = create_causal_mask(**mask_kwargs)

    # --- 返回所有计算出的、后续需要的值 ---
    return hybrid_attention_mask, position_ids, past_key_values, use_cache, cache_position

class SegQwenVL(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, 1) # 背景 / 前景 / 边缘
        self.model._create_hybrid_mask_and_dependencies = _create_hybrid_mask_and_dependencies.__get__(self)

    def pos_change(self):
        self.mask_token_id = self.config.mask_token_id
        self.seg_start_id = self.config.seg_start_id
        def get_rope_index_wrapper(model_self, *args, **kwargs):
            return get_rope_index_2_5(
                model_self,
                *args,
                seg_start_token_id=self.seg_start_id,
                seg_holder_token_id=self.mask_token_id,
                **kwargs
            )

        self.model.get_rope_index = types.MethodType(get_rope_index_wrapper, self.model)

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.FloatTensor = None, pixel_values: torch.FloatTensor = None,
                position_ids=None, labels: torch.LongTensor = None, do_classification: bool=False, output_hidden_states=False, **kwargs,):

        if self.mask_token_id is None:
            self.mask_token_id = self.config.mask_token_id
        if self.seg_start_id is None:
            self.seg_start_id = self.config.seg_start_id
        if do_classification:  ## 先假定单图分割。
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            image_embeds = self.model.get_image_features(pixel_values, kwargs['image_grid_thw'])
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            seg_mask = (input_ids == self.mask_token_id)

            ## abl!!!
            inputs_embeds[seg_mask] = inputs_embeds[seg_mask] + image_embeds[-seg_mask.sum():]
            kwargs.pop('inputs_embeds', None)
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
                dummy_logits = self.classifier(last_hidden_state) # 多进程训练防止卡住
                if hasattr(original_output, 'loss') and original_output.loss is not None:
                    dummy_loss = dummy_logits[0, 0].sum() * 0.0
                    original_output.loss += dummy_loss

            return original_output

