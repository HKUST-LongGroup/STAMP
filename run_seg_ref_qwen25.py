import argparse
import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
sys.path.append("/home/szhang/mllm/efficient_seg-main-cache")
from model.segment_anything import SamPredictor, sam_model_registry
from eval.utils import compute_logits_from_mask, show_points, masks_sample_points


import requests
from PIL import Image
from io import BytesIO
import re

from segment_predictor_qwen25 import GenerativeSegmenter
# from segment_predictor_cache import GenerativeSegmenter


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def upsample_tensor_vectorized(a, s):
    h, w = a.shape
    sh, sw = int(h * s), int(w * s)
    # Create an output tensor of zeros
    result = torch.zeros((sh, sw), dtype=a.dtype, device=a.device)
    # Calculate the target indices
    offset = int(s / 2)
    i_indices = torch.arange(h) * s + offset
    j_indices = torch.arange(w) * s + offset
    # Use broadcasting to fill the result tensor
    result[i_indices[:, None].long(), j_indices.long()] = a
    return result


def translate_sequence(sequence_str):
    """
    Translates a comma-separated sequence of categorical data_raw to numerical labels,
    identifying categories from the sequence.

    Parameters:
    sequence_str (str): The comma-separated sequence of categorical data_raw.

    Returns:
    list: The sequence of numerical labels.
    """
    # Split the string into a list of categories
    sequence = sequence_str.split('|')

    # strip the whitespace from each category
    sequence = [seq.strip() for seq in sequence]

    # Identify unique categories from the sequence
    unique_categories = list(dict.fromkeys(sequence))

    # place "others" at the beginning of the list
    if "others" in unique_categories:
        unique_categories.remove("others")
        unique_categories.insert(0, "others")

    # Create a dictionary to map each category to a unique integer
    category_to_label = {category: idx for idx, category in enumerate(unique_categories)}

    # Translate the sequence using the dictionary
    translated_sequence = [category_to_label[item] for item in sequence]

    return translated_sequence


def decode_mask(encoded_str):
    rows = encoded_str.strip("\n").split("\n ")
    decoded_list = []
    for row in rows:
        tokens = row.split("| ")
        for token in tokens:
            label, count = token.split(" *")
            decoded_list.extend([label] * int(count))
    return "|".join(decoded_list)


def run_model(args):
    # Model

    segmenter = GenerativeSegmenter(
        args.model_path,
        device_map="cuda",
        min_pixels=2048 * 28 * 28,
        max_pixels = 2500 * 28 * 28
    )
    sam_post_process = True

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_path)
    sam = sam.to(dtype=torch.float32, device='cuda')
    predictor = SamPredictor(sam)

    prompt_seg_single = args.query

    image_files = image_parser(args)
    images = load_images(image_files)
    image = images[0]
    w_ori, h_ori = image.size
    with torch.inference_mode():
        predictor.set_image(np.array(image))
        segmentation_masks, response_text = segmenter.generate_with_segmentation(image, prompt_seg_single)
    print(response_text)
    if segmentation_masks is None or len(segmentation_masks) == 0:
        print("No mask found.")

        return

    # assert len(segmentation_masks) == 1

    mask = segmentation_masks[0]

    mask_pred = pred_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).double(), size=(h_ori, w_ori), mode='nearest').squeeze(
        0).squeeze(0)

    new_mask_pred = np.zeros((mask_pred.shape[0], mask_pred.shape[1]))
    unique_classes = np.unique(mask_pred)
    new_mask_pred_sam = None
    if sam_post_process:
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
            new_mask_pred_sam = torch.from_numpy(sam_mask).to(pred_mask.device)
    else:
        new_mask_pred_sam = mask_pred
    # print(new_mask_pred_sam)
    mask_pred = mask_pred.unsqueeze(-1).repeat(1, 1, 3).numpy()
    # sam_mask = new_mask_pred
    #
    # sam_mask_s = sam_mask.astype("uint8")
    # sam_mask_s = Image.fromarray(sam_mask_s).convert('L')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(np.array(image)/255 * 0.7 + mask_pred*0.29)
    axes[0].set_title("Image")
    axes[0].axis('off')

    axes[1].imshow(new_mask_pred_sam)
    axes[1].set_title("Mask")
    axes[1].axis('off')

    plt.tight_layout()

    # save
    plt.savefig('/home/szhang/mllm/efficient_seg-main-cache/images/mask_2_5.png')
    print('done')
    plt.close()
    plt.axis('off')
    plt.imshow(new_mask_pred_sam, cmap='gray')
    # 不要边框
    plt.savefig('/home/szhang/mllm/efficient_seg-main-cache/images/new_mask_2_5.png', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/raid2/DATA/text4seg/STAMP_qwen_2.5")
    # parser.add_argument("--model-path", type=str, default="/apdcephfs_nj4/share_300377003/realzliu/uni_qwen_2b_new/checkpoint-3000/")
    # parser.add_argument("--model-path", type=str,
    #                     default="JiaZL/efficient_seg_gref")

    parser.add_argument("--image-file", type=str, default='/home/szhang/mllm/efficient_seg-main-cache/images/horses.png')
    parser.add_argument("--sam_path", type=str, default='/raid2/DATA/text4seg/vit/sam_vit_h_4b8939.pth')
    parser.add_argument("--query", type=str, default='Please segment the white horse in the image.')
    parser.add_argument("--sep", type=str, default=",")
    args = parser.parse_args()

    run_model(args)