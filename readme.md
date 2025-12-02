# Better, Stronger, Faster: Tackling the Trilemma in MLLM-based Segmentation with Simultaneous Textual Mask Prediction

<div align="center">

  <div>
      <a href="https://jiazhen-code.github.io/about.me/" target="_blank">Jiazhen Liu</a>,
      <a href="#" target="_blank">Mingkuan Feng</a>,
      <a href="https://zjuchenlong.github.io/" target="_blank">Long Chen</a>
  </div>
  <div>
      The Hong Kong University of Science and Technology (HKUST)
  </div>

  <br>

  <a href="https://arxiv.org/pdf/2512.00395" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-Coming%20Soon-inactive.svg?logo=arxiv&logoColor=b31b1b" alt="arXiv Paper">
  </a>
  <a href="https://huggingface.co/JiaZL/STAMP-2B-uni" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Project-FFD21E" alt="Hugging Face Project">
  </a>
  <img src="https://img.shields.io/badge/Demo-Coming%20Soon-inactive.svg?logo=gradio&logoColor=orange" alt="Demo Coming Soon">

  <br><br>
  <img src="https://i.imgur.com/waxVImv.png" alt="Teaser" width="100%">

</div>

## 📖 Introduction

**STAMP** (**S**imultaneous **T**extual **A**ll-**M**ask **P**rediction) is a novel MLLM-based segmentation paradigm that resolves the core “trilemma” in current methods: balancing **dialogue ability**, **segmentation performance**, and **inference speed**.

By decoupling autoregressive dialogue generation from non-autoregressive mask prediction, STAMP predicts the entire segmentation mask in a single forward pass parallel to the text response.

<details>
<summary>Click to view the Paradigm Comparison Figure</summary>
<p align="center">
  <img src="images/STAMP.png" width="80%">
</p>
</details>

## 🚀 Quick Start

### 1. Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/HKUST-LongGroup/STAMP.git
cd STAMP

# Create environment
conda create -n STAMP python=3.10
conda activate STAMP

# Install dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Download SAM checkpoint (Required for mask generation)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 2. Model Zoo

Currently, we have uploaded 2 versions of STAMP models to Hugging Face:

| Model Name | Paper Reference | Hugging Face | Description                                             |
| :--- | :--- | :--- |:--------------------------------------------------------|
| **STAMP-2B-uni** | **Table 5** | [🤗 Link](https://huggingface.co/JiaZL/STAMP-2B-uni) | Unified tasks (Dialogue and Segmentation), lightweight. |
| **STAMP-7B-lora** | **Table 2** (7B model) | [🤗 Link](https://huggingface.co/JiaZL/STAMP-7B-lora) | Higher capacity, best segmentation performance.         |


### 3. Inference

The code automatically downloads models from Hugging Face if not found locally.

#### Option A: Command Line Interface (CLI)

```bash
# Example with STAMP-2B
CUDA_VISIBLE_DEVICES="0" python run_seg_ref.py \
    --model-path "JiaZL/STAMP-2B-uni" \ 
    --image-file "images/horses.png" \
    --sam_path "HCMUE-Research/SAM-vit-h/sam_vit_h_4b8939.pth" \
    --query "Please segment the white horse in the image."
    
# For 7B variant, change --model-path to "JiaZL/STAMP-7B-lora"
```

#### Option B: Python API

```python
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image

# Import local modules
from segment_predictor_cache import GenerativeSegmenter
from model.segment_anything import sam_model_registry, SamPredictor
# [New] Import utility functions for SAM prompt generation
from eval.utils import compute_logits_from_mask, masks_sample_points

# --- Configuration ---
# Model paths
MODEL_PATH = "JiaZL/STAMP-2B-uni" 
SAM_PATH = "HCMUE-Research/SAM-vit-h/sam_vit_h_4b8939.pth"
IMAGE_PATH = "images/horses.png"
QUERY = "Please segment the white horse in the image."
USE_SAM = True  # Enable SAM refinement (Recommended: True)

# --- Load Models ---
print(f"Loading STAMP model from {MODEL_PATH}...")
segmenter = GenerativeSegmenter(
    MODEL_PATH,
    device_map="cuda",
    min_pixels=1024 * 28 * 28,
    max_pixels=1280 * 28 * 28
)

print(f"Loading SAM model from {SAM_PATH}...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_PATH)
sam = sam.to(dtype=torch.float32, device='cuda')
predictor = SamPredictor(sam)

# --- Inference ---
image = Image.open(IMAGE_PATH).convert("RGB")
w_ori, h_ori = image.size

with torch.inference_mode():
    # 1. Set SAM image embedding (Compute once for efficiency)
    if USE_SAM:
        predictor.set_image(np.array(image))
        
    # 2. Generate Coarse Mask using STAMP
    print("Generating coarse mask with STAMP...")
    segmentation_masks, response_text = segmenter.generate_with_segmentation(
        image, QUERY
    )
    print(f"Model Response: {response_text}")

    if not segmentation_masks or len(segmentation_masks) == 0:
        print("No mask generated.")
        exit()

    # Extract the first mask
    mask = segmentation_masks[0]

    # Resize coarse mask to original image size [H, W]
    mask_pred = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).double(),
        size=(h_ori, w_ori),
        mode='nearest'
    ).squeeze(0).squeeze(0)

    # --- SAM Refinement ---
    final_mask = np.zeros((h_ori, w_ori), dtype=np.float32)

    if USE_SAM:
        print("Refining mask with SAM...")
        # Get all unique class IDs (excluding background 0)
        unique_classes = torch.unique(mask_pred)
        
        for class_id in unique_classes:
            if class_id == 0: continue
            
            # Get binary mask for the current class
            binary_mask = (mask_pred == class_id).double().cpu()
            
            try:
                # Generate Prompts (Logits and Points) from the coarse mask
                logits = compute_logits_from_mask(binary_mask)
                point_coords, point_labels = masks_sample_points(binary_mask)
                
                # First pass prediction
                sam_mask, _, logit = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    mask_input=logits,
                    multimask_output=False
                )
                
                # Iterative refinement (Standard Cascade: 2 times)
                for _ in range(2):
                    sam_mask, _, logit = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        mask_input=logit,
                        multimask_output=False
                    )
                
                # Merge results into the final mask
                current_refined_mask = sam_mask[0].astype(np.float32)
                final_mask = np.maximum(final_mask, current_refined_mask)
                
            except Exception as e:
                print(f"SAM Error for class {class_id}: {e}")
                # Fallback to coarse mask if SAM fails
                final_mask = np.maximum(final_mask, binary_mask.numpy())
    else:
        # Use coarse mask directly if SAM is disabled
        final_mask = mask_pred.cpu().numpy()

# --- Save Result ---
# Convert to 0-255 uint8 format for saving
mask_uint8 = (final_mask > 0).astype(np.uint8) * 255

base_name = os.path.basename(IMAGE_PATH).split(".")[0]
save_name = f"{base_name}_mask_refined.png"

cv2.imwrite(save_name, mask_uint8)
print(f"Saved refined mask to {save_name}")
```

---

## 🖼️ Gallery & Showcases

STAMP is not only capable of standard referring segmentation but also excels in reasoning segmentation, VQA, and interactive multi-round conversation/segmentation.

Note, we **DO NOT** explicit train STAMP on multi-round data, 

### 1. Basic Capabilities
| Standard Ref-Seg | Reasoning Seg | Visual Question Answering |
| :---: | :---: | :---: |
| <img src="images/showcase1.png" width="100%"> | <img src="images/showcase2.png" width="100%"> | <img src="images/showcase3.png" width="100%"> |

### 2. Interactive Multi-Round Capabilities
STAMP can maintain context across multiple turns, follow incremental instructions, and seamlessly switch between dialogue and segmentation.

| Multi-round Dialogue | Multi-round Segmentation |
| :---: | :---: |
| <img src="images/showcase4.png" width="100%"> | <img src="images/showcase5.png" width="100%"> |

### 3. Unified Dialogue & Segmentation
<div align="center">
  <img src="images/showcase6.png" width="80%">
  <p><i>Examples of unified dialogue, explanation, and segmentation within the same interaction.</i></p>
</div>

---

## 📊 Evaluation & Training

### 1. Segmentation Evaluation
Evaluate Referring Expression Segmentation (RefCOCO/+/g, etc.):

```bash
bash scripts/eval_ref.sh
# Logs will be saved to: eval/eval_logs
```


### 2. VQA Evaluation
To evaluate VQA performance, you can directly use `lmm-eval`. 
**Note:** The weight and structural changes involved in STAMP **DO NOT** influence the standard VQA logic, ensuring general dialogue capabilities are preserved.

### 3. Training
We provide scripts for training both versions.

| Model Version | Training Script |
| :--- | :--- |
| **STAMP-2B** | `bash scripts/launch_all_2B.sh` |
| **STAMP-7B** | `bash scripts/launch_all_7B.sh` |

---

## 📂 Data Preparation

Please organize your datasets as follows in `playground/data`.

<details>
<summary><b>Click to expand Data Structure & Download Links</b></summary>

- Referring expression segmentation dataset
    - [RefCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip)
    - [RefCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip)
    - [RefCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip)
    - [RefCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip) ([saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip)) 

- Generalized referring expression segmentation dataset
  - [gRefCOCO](https://drive.google.com/drive/folders/1My2U6SuTAZG9yGBKe_PjsUJJgjdxOiiN)

- Reason Segmentation
  - [ReasonSeg](https://github.com/dvlab-research/LISA)

- [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)
    - COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
    - GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
    - OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)
    - TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
    - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

Download them from the above links, and organize them as follows.
```
├── playground/data
│   ├── refer_seg
│   │   ├── grefcoco
|   |       ├── grefs(unc).json
│   │   ├── images
|   |       ├── coco_2014
|   |       ├── saiapr_tc-12
│   │   ├── refclef
|   |       ├── instances.json
│   │   ├── refcoco
|   |       ├── instances.json
│   │       └── ...
│   │   ├── refcoco+
|   |       ├── instances.json
│   │       └── ...
│   │   └── refcocog
|   |       ├── instances.json
│   │       └── ...
│   ├── reason_seg
|   ├── coco
|   │   └── train2017
|   ├── gqa
│   |   └── images
|   ├── ocr_vqa
│   |   └── images
|   ├── textvqa
│   |   └── train_images
|   ├── vg
|   |    ├── VG_100K
|   |    └── VG_100K_2
|   └── llava_v1_5_mix665k.json
```

## Json files
Generate the json files:
```
python STAMP/data/create_refcoco_new.py
```
The processed JSON files are listed below:

* **Referring Expression Segmentation**
  * `STAMP/train/json_files/refclef_formatted_all_sentences_doubled_mp.json`
  * `STAMP/train/json_files/refcoco_formatted_all_sentences_doubled_mp.json`
  * `STAMP/train/json_files/refcoco+_formatted_all_sentences_doubled_mp.json`
  * `STAMP/train/json_files/refcocog_formatted_all_sentences_doubled_mp.json`

</details>


---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{stamp2024,
  title={Better, Stronger, Faster: Tackling the Trilemma in MLLM-based Segmentation with Simultaneous Textual Mask Prediction},
  author={Liu, Jiazhen and Feng, Mingkuan and Chen, Long},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

If you have any questions, please feel free to reach out at `jliugj@connect.ust.hk`.
