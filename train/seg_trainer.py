import torch
from torch import nn
from transformers import DynamicCache
from trl import SFTTrainer
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from segment_predictor import find_image_patch_info
from .utils import append_after_segment_torch
import torchvision.transforms as T

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

import os
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "-1") in ["-1", "0"]

class WeightedDiceBCELoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(WeightedDiceBCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        loss = self.alpha * BCE + self.beta * dice_loss

        return loss

class SegmentationSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-fetch token IDs for convenience
        self.seg_token_id = self.processing_class.tokenizer.convert_tokens_to_ids("<|seg|>")
        self.mask_token_id = self.processing_class.tokenizer.convert_tokens_to_ids("<|mask|>")
        self.yes_token_id = self.processing_class.tokenizer.convert_tokens_to_ids("<|yes|>")
        self.no_token_id = self.processing_class.tokenizer.convert_tokens_to_ids("<|no|>")
        self.image_pad_id = self.processing_class.tokenizer.convert_tokens_to_ids('<|image_pad|>')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        torch.cuda.empty_cache()
        # =================================================================
        # 1. COMPUTE STANDARD LANGUAGE MODELING LOSS
        # =================================================================

        # This call computes the standard cross-entropy loss on the 'text' field
        # It internally uses inputs['input_ids'] and inputs['labels']
        outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss_lm = outputs[0]

        # =================================================================
        # 2. COMPUTE SEGMENTATION LOSS (if applicable)
        # =================================================================
        loss_seg = torch.tensor(0.0, device=model.device)
        batch_size = inputs['input_ids'].shape[0]
        merge_size = self.processing_class.image_processor.merge_size
        all_seg_input = []
        all_seg_idx_mask = []
        all_seg_attn_mask = []
        all_pixel_values = []
        all_grid_thw = []
        start_img_num = 0
        pixel_values = inputs['pixel_values']
        for i in range(batch_size):
            input_ids = inputs['input_ids'][i]
            attn_masks = inputs['attention_mask'][i]

            img_num = (input_ids == self.image_pad_id).sum().item() * merge_size**2
            img_pixels = pixel_values[start_img_num: start_img_num + img_num]
            grid_thw = inputs['image_grid_thw'][i]
            start_img_num += img_num

            num_patches = find_image_patch_info(self.image_pad_id, input_ids)
            mask_query_ids = torch.full((num_patches,), self.mask_token_id, dtype=torch.long, device=input_ids.device)
            new_seg_input_ids, new_seg_attns = append_after_segment_torch(input_ids, attn_masks, self.seg_token_id, mask_query_ids)
            new_pixel_values = img_pixels.repeat([len(new_seg_input_ids), 1])
            grid_thw = grid_thw.repeat([len(new_seg_input_ids), 1])
            all_pixel_values.append(new_pixel_values)
            idx_mask = [ns == self.mask_token_id for ns in new_seg_input_ids]
            all_seg_input.extend(new_seg_input_ids)
            all_seg_idx_mask.extend(idx_mask)
            all_seg_attn_mask.extend(new_seg_attns)
            all_grid_thw.append(grid_thw)

        if len(all_seg_input) > 0:
            batch_grid_thw = torch.cat(all_grid_thw)
            batch_pixel_values = torch.cat(all_pixel_values, dim=0)
            batch_input_ids = rnn_utils.pad_sequence(
                all_seg_input,
                batch_first=True,
                padding_value=151643
            )

            batch_attn_mask = rnn_utils.pad_sequence(
                all_seg_attn_mask,
                batch_first=True,
                padding_value=0
            )

            batch_idx_mask = rnn_utils.pad_sequence(
                all_seg_idx_mask,
                batch_first=True,
                padding_value=False
            )

            seg_logits = model(input_ids=batch_input_ids, attention_mask=batch_attn_mask, pixel_values=batch_pixel_values,
                  image_grid_thw=batch_grid_thw, output_hidden_states=True, do_classification=True).bi_logits
            mask_preds = seg_logits[batch_idx_mask]

            mask_gt = inputs['masks']
            all_images = inputs['all_images']
            start_p = 0
            num_gt = 0

            for i in range(batch_size):
                gt_mask_list = mask_gt[i]
                for s, gt_mask in enumerate(gt_mask_list):
                    gt_mask = T.ToTensor()(gt_mask)[0].unsqueeze(0).unsqueeze(0).to(mask_preds.device)
                    h_bar, w_bar = batch_grid_thw[num_gt][1:]
                    h_bar, w_bar = h_bar / merge_size, w_bar / merge_size
                    h_bar, w_bar = h_bar.int().item(), w_bar.int().item()
                    gt_mask = F.interpolate(gt_mask.float(), size=(h_bar, w_bar), mode='bilinear', align_corners=False).squeeze(0).long()
                    num_p = int(h_bar * w_bar)
                    mask_pred = mask_preds[start_p: start_p + num_p]
                    start_p += num_p
                    gt_mask = gt_mask.view(-1)
                    binary_gt_labels = (gt_mask > 0.5).long()
                    num_pos = (binary_gt_labels == 1).sum().item()


                    loss_fn_weighted = WeightedDiceBCELoss(alpha=0.3, beta=0.7)
                    loss_seg_ = loss_fn_weighted(mask_pred.float(), binary_gt_labels.unsqueeze(1).float())
                    img_show = T.ToTensor()(all_images[i][-1]).permute(1, 2, 0).cpu()
                    loss_seg += loss_seg_
                    num_gt += 1

            if IS_MAIN_PROCESS:
                import matplotlib.pyplot as plt
                plt.subplot(121)
                hh, ww = img_show.shape[:2]
                pred_show = mask_pred.view(h_bar, w_bar).cpu().detach()
                pred_show = F.interpolate(pred_show.unsqueeze(0).unsqueeze(0).float(), size=(hh, ww), mode='nearest')[0][0]
                
                mask_rgb = (pred_show[..., None] > 0).repeat(1, 1, 3)
                
                img_show = img_show * 0.7 + mask_rgb * 0.3
                plt.imshow(img_show)
                plt.subplot(122)
                plt.imshow(binary_gt_labels.view(h_bar, w_bar).cpu().numpy())
                plt.savefig("seg_vis.png")
                plt.close()

            loss_seg = loss_seg / num_gt
        total_loss = loss_lm + loss_seg

        self._metrics['train']['loss_lm'].append(loss_lm.item())
        self._metrics['train']['loss_seg'].append(loss_seg.item())

        return (total_loss, outputs) if return_outputs else total_loss

    # --- [CORRECTED METHOD] ---
    def log(self, *args, **kwargs):
        """
        Override the log method to correctly calculate and display loss components.
        The fix is to REMOVE the .clear() calls.
        """
        # The 'logs' dictionary is usually the first positional argument.
        if args:
            logs = args[0]
        else:
            # Fallback for different calling conventions
            logs = kwargs

        # Only perform this logic on the main process and if there's something to log
        if self.state.is_local_process_zero and 'loss_lm' in self._metrics['train']:
            lm_losses = self._metrics['train']['loss_lm']
            seg_losses = self._metrics['train']['loss_seg']

            if len(lm_losses) > 0:
                # Calculate the average over the logging period
                mean_loss_lm = sum(lm_losses) / len(lm_losses)
                mean_loss_seg = sum(seg_losses) / len(seg_losses)

                # Update the logs dictionary with our calculated averages
                logs['loss_lm'] = round(mean_loss_lm, 4)
                logs['loss_seg'] = round(mean_loss_seg, 4)

                # CRITICAL: Overwrite the 'loss' key with the correct sum
                logs['loss'] = round(mean_loss_lm + mean_loss_seg, 4)

        # Call the original log method with the modified logs dictionary.
        # The parent method will handle the actual logging and will also
        # clear the self._metrics dictionary for the next logging interval.
        super().log(*args, **kwargs)