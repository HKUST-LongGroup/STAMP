<div align="center">
<h1>Better, Stronger, Faster: Tackling the Trilemma in MLLM-based Segmentation
with Simultaneous Textual Mask Prediction</h1>

<div>
    <a href='-' target='_blank'>-</a><sup>1</sup>&emsp;
    <a href='-' target='_blank'>-</a><sup>1</sup>&emsp;
    <a href='-' target='_blank'>-</a><sup>1</sup>&emsp;   
    <a href='-' target='_blank'>-</a><sup>2</sup>&emsp;
    <a href='-' target='_blank'>-</a><sup>2</sup>&emsp;
    <a href='-' target='_blank'>-</a><sup>3</sup>&emsp;
    <a href='-' target='_blank'>-</a><sup>3</sup>&emsp;
    <a href='-' target='_blank'>-</a><sup>3</sup>&emsp;
</div>
<div>
    <sup>1</sup>-; 
    <sup>2</sup>-; 
    <sup>3</sup>-;
</div>

[![Demo](https://img.shields.io/badge/Online-Demo-red)](#)
*(Coming soon)*
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](#)
*(Coming soon)*
[![Paper](https://img.shields.io/badge/arXiv-Paper-lightgrey.svg)](#)
*(To be released)*  

</div>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>


---

## Abstract
*Integrating segmentation into Multimodal Large Language Models (MLLMs) presents a core trilemma: simultaneously preserving dialogue ability, achieving high segmentation performance, and ensuring fast inference. Prevailing paradigms are forced into a compromise. Embedding prediction methods introduce a conflicting pixel-level objective that degrades the MLLM's general dialogue abilities. The alternative, next-token prediction, reframes segmentation as an autoregressive task, which preserves dialogue but forces a trade-off between poor segmentation performance with sparse outputs or prohibitive inference speeds with rich ones. We resolve this trilemma with **all-mask prediction**, a novel paradigm that decouples autoregressive dialogue generation from non-autoregressive mask prediction. We present *STAMP*: **S**imultaneous **T**extual **A**ll-**M**ask **P**rediction, an MLLM that embodies this paradigm. After generating a textual response, STAMP predicts an entire segmentation mask in a single forward pass by treating it as a parallel “fill-in-the-blank" task over image patches. This design maintains the MLLM's dialogue ability by avoiding conflicting objectives, enables high segmentation performance by leveraging rich, bidirectional spatial context for all mask tokens, and achieves exceptional speed. Extensive experiments show that STAMP significantly outperforms state-of-the-art methods across multiple segmentation benchmarks, providing a solution that excels in dialogue, segmentation, and speed without compromise.*
<div align="center">
  <img src="images/all_mask.jpg" width=80%>
  <div style="display: inline-block; color: #999; padding: 2px;">
      Comparison of MLLM-based segmentation paradigms. (a) Embedding Prediction: A conflicting pixel-level objective (e.g., LISA) degrades the MLLM's general dialogue capabilities. (b) Next-token Prediction: Generates masks autoregressively (e.g., Text4Seg), forcing a trade-off between poor segmentation performance (for sparse outputs) and slow inference (for rich outputs). (c) Our All-mask Prediction: We decouple dialogue generation (autoregressive) from mask generation (non-autoregressive). By simultaneously predicting all mask tokens as patch-wise classifications in a single pass, our paradigm resolves the segmentation trilemma, uniting preserved dialogue abilities, high segmentation performance and fast inference speed.
  </div>
</div>

---



## Dependencies and Installation

```
cd STAMP

# create new anaconda env
conda create -n STAMP python=3.10
conda activate STAMP

# install torch and dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

##  Project Structure

```text
STAMP/
├── data/                   # Data preprocessing and formatting scripts
│   ├── create_grefcoco.py
│   ├── create_refcoco_new.py
│   ├── grefer.py
│   └── ...
├── dataset/                # PyTorch dataset implementations
│   ├── grefer_seg_dataset.py
│   └── refer_seg_dataset.py
├── eval/                   # Evaluation scripts and metrics
│   ├── eval_refer_seg.py
│   ├── val_utils.py
│   └── ...
├── images/                 # Demo images
│   └── horses.png
├── model/                  # Model architecture definitions (STAMP, Qwen2-VL)
│   ├── segment_anything/
│   ├── modeling_qwen2_vl.py
│   └── qwen_changes.py
├── scripts/                # Shell scripts for running evaluation/training
│   ├── eval_ref.sh
|   └── launch_all.sh
├── train/                  # Training logic and entry points
│   ├── main_seg_train.py
│   ├── seg_trainer.py
│   ├── val_callback.py
│   └── ...
├── run_seg_ref.py          # Main inference/demo script
├── requirement.txt            
├── segment_predictor_cache.py # Predictor wrapper with cache
└── readme.md
```
## Datasets
- [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)
    - COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
    - GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
    - OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)
    - TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
    - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- Referring expression segmentation dataset
    - [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip)
    - [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip)
    - [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip)
    - [refCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip) ([saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip)) 
- Semantic segmentation dataset
    - [COCOStuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
    - [ADE20K](https://ade20k.csail.mit.edu/)
    - [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
    - [PASCAL Context](https://cs.stanford.edu/~roozbeh/pascal-context/)

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
│   ├── semantic_seg
│   │   ├── ADE20K
│   │       ├── annotations
│   │       └── images
│   │   ├── PAS20
│   │       ├── JPEGImages
│   │       ├── SegmentationClass
│   │       └── val.txt
│   │   └── PC59
│   │       ├── JPEGImages
│   │       ├── SegmentationClassContext
│   │       └── pascalcontext_val.txt
|   ├── coco
|   │   └── train2017
|   ├── cocostuff
|   │   └── annotations
|   |       ├── 000000000009_labelTrainIds.png
|   |       └── ...
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
To evaluate the VQA performance, you can directly evaluate it through `lmm-eval`. The weight and structural changes involved in the `STAMP` **DO NOT** influence the VQA logic.

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


## Quick Inference
### Quick Inference with STAMP-2B-uni
To run inference with STAMP-2B-uni, please pre-download the following checkpoints to expedite the workflow:
* **[STAMP-2B-uni model](https://huggingface.co/JiaZL/STAMP-2B-uni)**
* **[SAM checkpoint (sam_vit_h_4b8939.pth)](https://huggingface.co/HCMUE-Research/SAM-vit-h/blame/main/sam_vit_h_4b8939.pth)**
```
CUDA_VISIBLE_DEVICES="0" python STAMP/run_seg_ref.py --model-path="JiaZL/STAMP-2B-uni" --image-file="STAMP/images/horses.png" --sam_path="sam_vit_h_4b8939.pth" --query="Please segment the trees in the image."
```
### Quick Inference with STAMP-7B
To run inference with STAMP-7B model, please pre-download the following checkpoints to expedite the workflow:
* **[STAMP-7B model](https://huggingface.co/JiaZL/STAMP-7B-lora)**
* **[SAM checkpoint (sam_vit_h_4b8939.pth)](https://huggingface.co/HCMUE-Research/SAM-vit-h/blame/main/sam_vit_h_4b8939.pth)**
```
CUDA_VISIBLE_DEVICES="0" python STAMP/run_seg_ref.py --model-path="JiaZL/seg-7B" --image-file="STAMP/images/horses.png" --sam_path="sam_vit_h_4b8939.pth" --query="Please segment the trees in the image."
```

## Model evaluation
Referring expression segmengtation:
```
bash STAMP/scripts/eval_ref.sh
```
### Evaluation logs
The evaluation logs are saved in the directory: STAMP/eval/eval_logs


## Model training
### Train STAMP-2B
```
bash STAMP/scripts/launch_all_2B.sh
```
### Train STAMP-7B
```
bash STAMP/scripts/launch_all_7B.sh
```
### Training logs
To be released.
## Experimental results

<p align="center"> <img src="images/results1.jpg" width="80%"> </p>
<p align="center"> <img src="images/results2.jpg" width="40%"> </p>
<p align="center"> <img src="images/results3.jpg" width="40%"> </p>
<p align="center"> <img src="images/results4.jpg" width="80%"> </p>


## Showcases of STAMP

**1. Standard Referring Segmentation**
<p align="center"> <img src="images/showcase1.png" width="80%"> </p>

**2. Reasoning Segmentation**
<p align="center"> <img src="images/showcase2.png" width="50%"> </p>

**3. Visual Question Answering**
<p align="center"> <img src="images/showcase3.png" width="50%"> </p>

**4. Multi-round Dialogue**
<p align="center"> <img src="images/showcase4.png" width="50%"> </p>

**5. Multi-round Segmentation**
<p align="center"> <img src="images/showcase5.png" width="50%"> </p>

**6. Unified Dialogue & Segmentation Examples**
<p align="center"> <img src="images/showcase6.png" width="60%"> </p>



## Citation

## License

## Acknowledgement


## Contact
If you have any questions, please feel free to reach out at `jliugj@connect.ust.hk`.


