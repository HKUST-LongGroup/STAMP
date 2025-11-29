import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from PIL import Image
from .utils import AverageMeter, Summary, intersectionAndUnionGPU, compute_logits_from_mask, masks_sample_points


def run_in_process_evaluation(model, accelerator, eval_dataloader, sam_predictor=None):
    # --- 1. Initialize all metric recorders ---
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    use_sam = sam_predictor is not None
    if use_sam:
        intersection_meter_sam = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter_sam = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter_sam = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    progress_bar = tqdm(eval_dataloader, disable=not accelerator.is_main_process, desc="Running Evaluation")

    # --- 2. Iterate over the evaluation dataset ---
    for batch in progress_bar:
        images, masks, image_names, questions, image_paths = batch
        image, gt_masks, image_name, prompts = images[0], masks[0], image_names[0], questions[0]
        w_ori, h_ori = image.size

        total_intersection = torch.zeros(2, device=accelerator.device)
        total_union = torch.zeros(2, device=accelerator.device)
        total_acc_iou = torch.zeros(2, device=accelerator.device)

        if use_sam:
            total_intersection_sam = torch.zeros(2, device=accelerator.device)
            total_union_sam = torch.zeros(2, device=accelerator.device)
            total_acc_iou_sam = torch.zeros(2, device=accelerator.device)

        num_masks_in_image = len(prompts)
        if num_masks_in_image == 0:
            continue

        with torch.inference_mode():
            if use_sam:
                predictor = sam_predictor
                predictor.set_image(np.array(image))

            for i, question in enumerate(prompts):
                gt_mask = gt_masks[i].to(accelerator.device)

                # --- Key difference: directly use the passed-in model object for inference ---
                segmentation_masks, _ = model.generate_with_segmentation(image, question)

                if segmentation_masks is None or len(segmentation_masks) == 0:
                    pred_mask = torch.zeros((h_ori, w_ori), device=accelerator.device)
                else:
                    mask = segmentation_masks[0].to(accelerator.device)
                    pred_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).double(), size=(h_ori, w_ori),
                                              mode='nearest').squeeze()

                # --- SAM post-processing (if enabled) ---
                if use_sam:
                    sam_refined_mask = torch.zeros((h_ori, w_ori), device=accelerator.device)
                    unique_classes = torch.unique(pred_mask)
                    for class_id in unique_classes:
                        if class_id == 0: continue
                        binary_mask = (pred_mask == class_id).double()
                        try:
                            logits = compute_logits_from_mask(binary_mask)
                            point_coords, point_labels = masks_sample_points(binary_mask)
                            sam_mask, _, logit = predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                                                   mask_input=logits, multimask_output=False)
                            for _ in range(2):
                                sam_mask, _, logit = predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                                                       mask_input=logit, multimask_output=False)
                        except Exception:
                            sam_mask = np.zeros((1, h_ori, w_ori))
                        sam_refined_mask[torch.from_numpy(sam_mask[0] > 0).to(accelerator.device)] = class_id

                # --- Metric calculation (current mask) ---
                # Use the original model for prediction
                # <-- MODIFICATION 1: Add ignore_index=255
                intersection_i, union_i, _ = intersectionAndUnionGPU(pred_mask, gt_mask, 2, ignore_index=255)
                total_intersection += intersection_i
                total_union += union_i
                # <-- MODIFICATION 2: Change epsilon to 1e-5
                iou_per_sample = intersection_i / (union_i + 1e-5)
                iou_per_sample[union_i == 0] = 1.0
                total_acc_iou += iou_per_sample

                # SAM optimized prediction
                if use_sam:
                    # <-- MODIFICATION 1 (SAM): Add ignore_index=255
                    intersection_sam_i, union_sam_i, _ = intersectionAndUnionGPU(sam_refined_mask, gt_mask, 2, ignore_index=255)
                    total_intersection_sam += intersection_sam_i
                    total_union_sam += union_sam_i
                    # <-- MODIFICATION 2 (SAM): Change epsilon to 1e-5
                    iou_per_sample_sam = intersection_sam_i / (union_sam_i + 1e-5)
                    iou_per_sample_sam[union_sam_i == 0] = 1.0
                    total_acc_iou_sam += iou_per_sample_sam

        # Update global recorders
        intersection_meter.update(total_intersection.cpu().numpy())
        union_meter.update(total_union.cpu().numpy())
        acc_iou_meter.update(total_acc_iou.cpu().numpy(), n=num_masks_in_image)
        if use_sam:
            intersection_meter_sam.update(total_intersection_sam.cpu().numpy())
            union_meter_sam.update(total_union_sam.cpu().numpy())
            acc_iou_meter_sam.update(total_acc_iou_sam.cpu().numpy(), n=num_masks_in_image)

    # --- 3. Aggregate metrics from all GPUs ---
    all_intersections = accelerator.gather_for_metrics(torch.from_numpy(intersection_meter.sum).to(accelerator.device))
    all_unions = accelerator.gather_for_metrics(torch.from_numpy(union_meter.sum).to(accelerator.device))
    all_giou_sum = accelerator.gather_for_metrics(torch.from_numpy(acc_iou_meter.sum).to(accelerator.device))
    all_giou_count = accelerator.gather_for_metrics(torch.tensor(acc_iou_meter.count, device=accelerator.device))

    if use_sam:
        all_intersections_sam = accelerator.gather_for_metrics(
            torch.from_numpy(intersection_meter_sam.sum).to(accelerator.device))
        all_unions_sam = accelerator.gather_for_metrics(torch.from_numpy(union_meter_sam.sum).to(accelerator.device))
        all_giou_sum_sam = accelerator.gather_for_metrics(
            torch.from_numpy(acc_iou_meter_sam.sum).to(accelerator.device))
        all_giou_count_sam = accelerator.gather_for_metrics(
            torch.tensor(acc_iou_meter_sam.count, device=accelerator.device))

    # --- 4. Calculate final results and return on the main process ---
    final_metrics = {}
    if accelerator.is_main_process:
        # original model metrics
        # <-- MODIFICATION 2 (cIoU): Change epsilon to 1e-5
        iou_class = torch.sum(all_intersections, dim=0) / (torch.sum(all_unions, dim=0) + 1e-5)
        ciou = iou_class[1].item()
        giou_sum = torch.sum(all_giou_sum, dim=0)[1]
        giou_count = torch.sum(all_giou_count)
        giou = (giou_sum / giou_count).item() if giou_count > 0 else 0.0
        final_metrics['giou'] = giou
        final_metrics['ciou'] = ciou

        # SAM optimized metrics
        if use_sam:
            # <-- MODIFICATION 2 (cIoU SAM): Change epsilon to 1e-5
            iou_class_sam = torch.sum(all_intersections_sam, dim=0) / (torch.sum(all_unions_sam, dim=0) + 1e-5)
            ciou_sam = iou_class_sam[1].item()
            giou_sum_sam = torch.sum(all_giou_sum_sam, dim=0)[1]
            giou_count_sam = torch.sum(all_giou_count_sam)
            giou_sam = (giou_sum_sam / giou_count_sam).item() if giou_count_sam > 0 else 0.0
            final_metrics['sam_giou'] = giou_sam
            final_metrics['sam_ciou'] = ciou_sam

    return final_metrics