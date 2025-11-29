#!/bin/bash

# ==============================================================================
# Script Description:
#   This script is used to evaluate the performance of the GenerativeSegmenter model on all specified Refer-Seg datasets.
#   All path parameters are specified within the script.
# ==============================================================================

# --- CONFIGURATION (All paths are specified here) ---
MODEL_PATH="STAMP-2B-uni/"
SAM_PATH="sam_vit_h_4b8939.pth"
IMAGE_FOLDER="playground/data/refer_seg"

# --- List of datasets to evaluate ---
DATASET_SPLITS=(
    "refcoco|unc|val"
    "refcoco|unc|testA"
    "refcoco|unc|testB"
    "refcoco+|unc|val"
    "refcoco+|unc|testA"
    "refcoco+|unc|testB"
    "refcocog|umd|val"
    "refcocog|umd|test"
)

# SPLIT_OPTIONS=("grefcoco|unc|testA" "grefcoco|unc|testB" "grefcoco|unc|val")

# --- Start evaluation loop ---
echo "=================================================="
echo "Starting evaluation for model: $MODEL_PATH"
echo "=================================================="

# Iterate over all datasets and execute evaluation commands
for split in "${DATASET_SPLITS[@]}"; do
    echo ""
    echo "--- Evaluating dataset: $split ---"

    # Construct a unique output directory name, e.g., "output_eval/refcoco_unc_val/"
    OUTPUT_DIR="output_eval/${split//|/_}/"

    accelerate launch --num_processes=4 --gpu_ids "0,1,6,7" STAMP/eval/eval_refer_seg.py \
        --model_path "$MODEL_PATH" \
        --sam_path "$SAM_PATH" \
        --image_folder "$IMAGE_FOLDER" \
        --dataset_split "$split" \
        --save_file "$OUTPUT_DIR" \
#        --save_masks

done

echo ""
echo "All dataset evaluations completed!"