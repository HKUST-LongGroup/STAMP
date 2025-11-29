import torch

def append_after_segment_torch(input_ids, attn_masks, seg_id, new_ids):
    if not isinstance(input_ids, torch.Tensor):
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    else:
        input_ids_t = input_ids.to(torch.long)

    if not isinstance(attn_masks, torch.Tensor):
        attn_masks_t = torch.tensor(attn_masks, dtype=torch.long)
    else:
        attn_masks_t = attn_masks.to(torch.long)

    if not isinstance(new_ids, torch.Tensor):
        new_ids_t = torch.tensor(new_ids, dtype=torch.long)
    else:
        new_ids_t = new_ids.to(torch.long)

    seg_indices = torch.where(input_ids_t == seg_id)[0]

    result_ids_list = []
    result_masks_list = []
    for idx in seg_indices:
        split_point = idx.item() + 1
        prefix_ids = input_ids_t[:split_point]
        new_sequence_ids = torch.cat((prefix_ids, new_ids_t), dim=0)
        result_ids_list.append(new_sequence_ids)
        prefix_mask = attn_masks_t[:split_point]
        new_mask_segment = torch.ones_like(new_ids_t)
        new_sequence_mask = torch.cat((prefix_mask, new_mask_segment), dim=0)
        result_masks_list.append(new_sequence_mask)


    return result_ids_list, result_masks_list