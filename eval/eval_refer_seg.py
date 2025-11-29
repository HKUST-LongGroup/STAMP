import argparse
import torch
import os
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
import torch.nn.functional as F
import datetime
from model.segment_anything import SamPredictor, sam_model_registry
from dataset.refer_seg_dataset import ValDataset
from dataset.grefer_seg_dataset import grefcocoValDataset
from data.question_answer_list import QUESTION_PARTIAL
from segment_predictor import GenerativeSegmenter
from eval.utils import AverageMeter, Summary, intersectionAndUnionGPU, \
    compute_logits_from_mask, masks_sample_points

from torch.utils.data import Dataset, DataLoader
import math

# --- Accelerate Import ---
from accelerate import Accelerator


def get_chunk(ds, n, k):
    chunk_size = math.ceil(len(ds) / n)
    i = chunk_size * k
    start_index = i
    end_index = i + chunk_size
    ds.refer_seg_ds["images"] = ds.refer_seg_ds["images"][start_index:end_index]
    return ds


def gget_chunk(ds, n, k):
    chunk_size = math.ceil(len(ds) / n)
    i = chunk_size * k
    start_index = i
    end_index = i + chunk_size
    ds.loaded_images = ds.loaded_images[start_index:end_index]
    return ds


class CustomDataset(Dataset):
    def __init__(self, sub_dataset):
        self.dataset = sub_dataset

    def __getitem__(self, index):
        image, masks, questions, image_path = self.dataset[index]
        image_name = os.path.basename(image_path).split(".")[0]
        questions = [random.choice(QUESTION_PARTIAL).replace("[class_name]", q) for q in questions]
        return image, masks, image_name, questions, image_path

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    images, masks, image_names, questions, image_paths = zip(*batch)
    return images, masks, image_names, questions, image_paths


def create_data_loader(args, sub_dataset, batch_size=1):
    assert batch_size == 1, "Batch size must be 1 for this evaluation script."
    dataset = CustomDataset(sub_dataset)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn)


def eval_model(args):
    # --- Initialize Accelerator ---
    accelerator = Accelerator()

    # --- Model and Predictor Initialization ---
    segmenter = GenerativeSegmenter(args.model_path, device_map=accelerator.device, min_pixels=args.min_pixels,
                                    max_pixels=args.max_pixels)

    sam_help = args.sam_path is not None
    if sam_help:
        sam = sam_model_registry["vit_h"](checkpoint=args.sam_path)
        sam = sam.to(dtype=torch.float32, device=accelerator.device)
        predictor = SamPredictor(sam)
    else:
        predictor = None

    # --- Dataset and DataLoader Initialization ---
    if accelerator.is_main_process:
        print("Loading dataset...")

    # First, load the full dataset based on the parameters
    if "grefcoco" in args.dataset_split:
        val_dataset = grefcocoValDataset(args.image_folder, args.dataset_split)
    else:
        val_dataset = ValDataset(args.image_folder, args.dataset_split)

    if accelerator.is_main_process:
        total_data_size = len(val_dataset)
        print(f"Total evaluation data volume (full dataset): {total_data_size} samples.")

    # Then, get a chunk of the dataset as needed
    if "grefcoco" in args.dataset_split:
        sub_dataset = gget_chunk(val_dataset, args.num_chunks, args.chunk_idx)
    else:
        sub_dataset = get_chunk(val_dataset, args.num_chunks, args.chunk_idx)

    data_loader = create_data_loader(args, sub_dataset)
    data_loader = accelerator.prepare(data_loader)

    # --- Metric Meters Initialization ---
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    if sam_help:
        intersection_meter_sam = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter_sam = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter_sam = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    progress_bar = tqdm(data_loader, disable=not accelerator.is_main_process, total=len(data_loader))

    for batch in progress_bar:
        images, masks, image_names, questions, image_paths = batch
        image, gt_masks, image_name, prompts = images[0], masks[0], image_names[0], questions[0]
        w_ori, h_ori = image.size

        total_intersection = torch.zeros(2, device=accelerator.device)
        total_union = torch.zeros(2, device=accelerator.device)
        total_acc_iou = torch.zeros(2, device=accelerator.device)

        if sam_help:
            total_intersection_sam = torch.zeros(2, device=accelerator.device)
            total_union_sam = torch.zeros(2, device=accelerator.device)
            total_acc_iou_sam = torch.zeros(2, device=accelerator.device)

        num_masks_in_image = len(prompts)

        with torch.inference_mode():
            if sam_help:
                predictor.set_image(np.array(image))

            for i, question in enumerate(prompts):
                gt_mask = gt_masks[i].to(accelerator.device).float().contiguous()
                segmentation_masks, _ = segmenter.generate_with_segmentation(image, question)

                if segmentation_masks is None or len(segmentation_masks) == 0:
                    pred_mask = torch.zeros((h_ori, w_ori), device=accelerator.device)
                else:
                    mask = segmentation_masks[0].to(accelerator.device)
                    pred_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).double(), size=(h_ori, w_ori),
                                              mode='nearest').squeeze()

                # if accelerator.is_main_process:
                #     print("\n" + "=" * 20 + " DEBUG INFO (first iteration) " + "=" * 20)
                #     print(f"Image Name: {image_name}")
                #     print(f"GT Mask Shape: {gt_mask.shape}")
                #     print(f"GT Mask DType: {gt_mask.dtype}")
                #     print(f"Unique values in GT Mask: {torch.unique(gt_mask)}")
                #     print(f"Pred Mask Shape: {pred_mask.shape}")
                #     print(f"Pred Mask DType: {pred_mask.dtype}")
                #     # Print the number of non-zero pixels in the predicted mask to check if the model generated an all-black image
                #     print(f"Number of non-zero pixels in Pred Mask: {torch.count_nonzero(pred_mask)}")
                #     print("=" * 68)

                sam_refined_mask = torch.zeros_like(pred_mask)
                if sam_help:
                    unique_classes = torch.unique(pred_mask)
                    for class_id in unique_classes:
                        if class_id == 0: continue
                        binary_mask = (pred_mask == class_id).double().cpu()
                        try:
                            logits = compute_logits_from_mask(pred_mask.cpu())
                            point_coords, point_labels = masks_sample_points(binary_mask)
                            sam_mask, _, logit = predictor.predict(point_coords=point_coords,
                                                                   point_labels=point_labels,
                                                                   mask_input=logits, multimask_output=False)
                            for _ in range(2):
                                sam_mask, _, logit = predictor.predict(point_coords=point_coords,
                                                                       point_labels=point_labels,
                                                                       mask_input=logit, multimask_output=False)
                            sam_mask = sam_mask[0].astype(np.float32)
                        except Exception as E:
                            print(f"Error: {E}")
                            sam_mask = np.zeros((h_ori, w_ori))
                        sam_refined_mask = torch.from_numpy(sam_mask).to(accelerator.device)
                        # sam_refined_mask[torch.from_numpy(sam_mask[0] > 0).to(accelerator.device)] = class_id

                intersection_i, union_i, _ = intersectionAndUnionGPU(pred_mask, gt_mask, 2, ignore_index=255)

                total_intersection += intersection_i
                total_union += union_i

                iou_per_sample = intersection_i / (union_i + 1e-5)
                iou_per_sample[union_i == 0] = 1.0
                total_acc_iou += iou_per_sample

                if sam_help:
                    intersection_sam_i, union_sam_i, _ = intersectionAndUnionGPU(sam_refined_mask, gt_mask, 2,
                                                                                 ignore_index=255)
                    total_intersection_sam += intersection_sam_i
                    total_union_sam += union_sam_i

                    iou_per_sample_sam = intersection_sam_i / (union_sam_i + 1e-5)
                    iou_per_sample_sam[union_sam_i == 0] = 1.0
                    total_acc_iou_sam += iou_per_sample_sam

                if args.save_masks and accelerator.is_main_process:
                    ds_split_sanitized = args.dataset_split.replace("|", "_")
                    model_name = os.path.basename(args.model_path.strip('/'))
                    save_path = os.path.join(args.save_file, model_name, ds_split_sanitized, "masks", image_name)
                    if not os.path.exists(save_path): os.makedirs(save_path)

                    Image.fromarray(pred_mask.cpu().numpy().astype("uint8") * 255).convert('L').save(
                        os.path.join(save_path, f"{i}_pred_mask.png"))
                    if sam_help:
                        Image.fromarray(sam_refined_mask.cpu().numpy().astype("uint8") * 255).convert('L').save(
                            os.path.join(save_path, f"{i}_sam_mask.png"))
                    Image.fromarray(gt_mask.cpu().numpy().astype("uint8") * 255).convert('L').save(
                        os.path.join(save_path, f"{i}_gt_mask.png"))
                    image.save(os.path.join(save_path, f"{i}_image.png"))

        intersection_meter.update(total_intersection.cpu().numpy())
        union_meter.update(total_union.cpu().numpy())
        if sam_help:
            intersection_meter_sam.update(total_intersection_sam.cpu().numpy())
            union_meter_sam.update(total_union_sam.cpu().numpy())
        if num_masks_in_image > 0:
            total_acc_iou = total_acc_iou / num_masks_in_image
            acc_iou_meter.update(total_acc_iou.cpu().numpy(), n=num_masks_in_image)
            if sam_help:
                total_acc_iou_sam = total_acc_iou_sam / num_masks_in_image
                acc_iou_meter_sam.update(total_acc_iou_sam.cpu().numpy(), n=num_masks_in_image)
        # break
    # --- Synchronize metrics across all processes ---
    all_intersections = accelerator.gather_for_metrics(torch.from_numpy(intersection_meter.sum).to(accelerator.device))
    all_unions = accelerator.gather_for_metrics(torch.from_numpy(union_meter.sum).to(accelerator.device))
    all_giou_sum = accelerator.gather_for_metrics(torch.from_numpy(acc_iou_meter.sum).to(accelerator.device))
    all_giou_count = accelerator.gather_for_metrics(torch.tensor(acc_iou_meter.count, device=accelerator.device))

    all_intersections = all_intersections.view(-1, 2)
    all_unions = all_unions.view(-1, 2)
    all_giou_sum = all_giou_sum.view(-1, 2)
    all_giou_count = all_giou_count.view(-1, 1)

    if sam_help:
        all_intersections_sam = accelerator.gather_for_metrics(
            torch.from_numpy(intersection_meter_sam.sum).to(accelerator.device))
        all_unions_sam = accelerator.gather_for_metrics(torch.from_numpy(union_meter_sam.sum).to(accelerator.device))
        all_giou_sum_sam = accelerator.gather_for_metrics(
            torch.from_numpy(acc_iou_meter_sam.sum).to(accelerator.device))
        all_giou_count_sam = accelerator.gather_for_metrics(
            torch.tensor(acc_iou_meter_sam.count, device=accelerator.device))

        all_intersections_sam = all_intersections_sam.view(-1, 2)
        all_unions_sam = all_unions_sam.view(-1, 2)
        all_giou_sum_sam = all_giou_sum_sam.view(-1, 2)
        all_giou_count_sam = all_giou_count_sam.view(-1, 1)


    # --- Only calculate and output final results on the main process ---
    if accelerator.is_main_process:
        iou_class = torch.sum(all_intersections, dim=0) / (torch.sum(all_unions, dim=0) + 1e-5)
        # print(all_intersections, all_unions, iou_class)
        ciou = iou_class[1].item()
        giou = (torch.sum(all_giou_sum, dim=0)[1] / torch.sum(all_giou_count)).item()

        if sam_help:
            iou_class_sam = torch.sum(all_intersections_sam, dim=0) / (torch.sum(all_unions_sam, dim=0) + 1e-5)
            ciou_sam = iou_class_sam[1].item()
            giou_sam = (torch.sum(all_giou_sum_sam, dim=0)[1] / torch.sum(all_giou_count_sam)).item()
        else:
            giou_sam, ciou_sam = 0.0, 0.0

        # <--- Added: Calculate and print accurate evaluation totals ---
        total_evaluated_images = len(sub_dataset)  # Total images evaluated
        total_evaluated_masks = torch.sum(all_giou_count).item()  # Total masks/prompts evaluated
        # <--- End added ---

        print("\n" + "=" * 50)
        print(f"Evaluation finished for: {args.model_path}")
        print(f"Dataset: {args.dataset_split}")
        print("-" * 50)
        # <--- Added: Print evaluation sample counts ---
        print(f"Total images evaluated: {total_evaluated_images}")
        print(f"Total masks/prompts evaluated: {total_evaluated_masks}")
        print("-" * 50)
        # <--- End added ---
        print(f"Raw Model Mask   -> gIoU: {giou:.4f}, cIoU: {ciou:.4f}")
        if sam_help:
            print(f"SAM-Refined Mask -> gIoU: {giou_sam:.4f}, cIoU: {ciou_sam:.4f}")
        print("=" * 50 + "\n")

        # --- Dynamically construct output file path and write results ---
        model_name = os.path.basename(args.model_path.strip('/'))
        ds_split_sanitized = args.dataset_split.replace("|", "_")
        output_dir = os.path.join(args.save_file, model_name, ds_split_sanitized)
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, "evaluation_results.txt")

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chunk_info = f"(chunk {args.chunk_idx + 1}/{args.num_chunks})" if args.num_chunks > 1 else ""

        header_text = f"[{current_time}] Model: {args.model_path}, Dataset: {args.dataset_split} {chunk_info}\n"
        # <--- Added: Also record evaluation sample counts in the file ---
        eval_stats_text = f"  - Evaluated on {total_evaluated_images} images and {total_evaluated_masks} masks.\n"
        # <--- End added ---
        output_text = f"  - Raw Model Mask   -> gIoU: {giou:.4f}, cIoU: {ciou:.4f}\n"
        if sam_help:
            output_text += f"  - SAM-Refined Mask -> gIoU: {giou_sam:.4f}, cIoU: {ciou_sam:.4f}\n"

        with open(output_filepath, "a") as file:
            file.write(header_text)
            file.write(eval_stats_text) # <--- Added
            file.write(output_text)
            file.write("-" * 60 + "\n")

        print(f"Results appended to: {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/raid2/DATA/text4seg/model_trained_qwen_2b/',
                        help="Path to your GenerativeSegmenter model checkpoint.")
    parser.add_argument("--sam_path", type=str, default='/efficient_sag4text/sam_vit_h_4b8939.pth', help="Path to the SAM checkpoint.")
    parser.add_argument("--image_folder", type=str, default='/efficient_sag4text/seg_data/refer_seg', help="Root folder for the dataset images.")
    parser.add_argument("--dataset_split", type=str, default="refcoco|unc|val", help="Dataset split to evaluate on.")
    parser.add_argument("--save_file", type=str, default="output_eval_accelerated/",
                        help="Root directory to save evaluation outputs (masks and metrics).")
    parser.add_argument("--save_masks", action='store_true', help="Set this flag to save output masks and images.")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--min_pixels", type=int, default=1024*28 * 28, help="Minimum pixels for segmentation.")
    parser.add_argument("--max_pixels", type=int, default=1024*28 * 28, help="Maximum pixels for segmentation.")
    args = parser.parse_args()

    eval_model(args)