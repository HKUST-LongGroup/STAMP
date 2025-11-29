import os
import numpy as np
from PIL import Image
import random
from pycocotools import mask
from pycocotools.coco import COCO
from refer import REFER
import matplotlib.pyplot as plt
from question_answer_list import QUESTION_ALL, QUESTION_PARTIAL, QUESTION_CONDITION, ANSWER_ALL, ANSWER_PARTIAL, \
    ANSWER_CONDITION
from utils import encode_mask
import json
import uuid
import multiprocessing as mp
from tqdm import tqdm  # <--- Step 1: Import tqdm

# --- Settings ---
data_path = "/apdcephfs_qy4/share_302593112/realzliu/dataset_open/lisa_dataset/refer_seg"
ds = "refclef" # refcoco, refcoco+, refcocog, refclef
mask_save_dir = f"/efficient_sag4text/playground/data/masks/{ds}/"
os.makedirs(mask_save_dir, exist_ok=True)
if ds == "refcocog":
    splitBy = "umd"
else:
    splitBy = "unc"

# --- SOLUTION PART 1: Create an initializer for worker processes ---
# This global variable will hold the REFER object for each worker process
refer_api = None


def init_worker(d_path, d_set, s_by):
    """
    This function is called once per worker process.
    It initializes the expensive REFER object and stores it in a global variable
    that is local to this specific process.
    """
    global refer_api
    # print(f"Process {os.getpid()} initializing REFER object...") # You can comment out this line to avoid cluttering the output
    refer_api = REFER(d_path, d_set, s_by)
    # print(f"Process {os.getpid()} initialization complete.") # You can comment out this line to avoid cluttering the output


# --- The worker function, now much faster ---
def process_image(image_info):
    """
    Processes a single image. It now USES the pre-initialized
    refer_api object instead of creating a new one.
    """
    global refer_api

    refs = refer_api.imgToRefs.get(image_info["id"], [])
    if not refs:
        return None

    item = {}

    if ds == "refclef":
        image_path = os.path.join("refer_seg/images/saiapr_tc-12", image_info["file_name"])
    else:
        image_path = os.path.join("refer_seg/images/coco_2014/train2014", image_info["file_name"])
    item["images"] = [image_path]

    mask_paths = []
    messages = []
    refs_to_process = [random.choice(refs) for _ in range(2)]

    for round_num, ref in enumerate(refs_to_process):
        sentences = ref['sentences']
        ann = refer_api.refToAnn[ref['ref_id']]

        if isinstance(ann["segmentation"][0], list):
            rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
        else:
            rle = ann["segmentation"]

        m = mask.decode(rle)
        if m.ndim == 3:
            m = np.sum(m, axis=2)
        m[m > 0] = 1

        mask_gt = Image.fromarray(m.astype(np.uint8) * 255)
        mask_filename = f"{image_info['id']}_{ref['ref_id']}_{uuid.uuid4()}.png"
        mask_path = os.path.join(mask_save_dir, mask_filename)
        mask_gt.save(mask_path)
        mask_paths.append(mask_path)

        sentence = random.choice(sentences)['sent']
        question = random.choice(QUESTION_PARTIAL).replace("[class_name]", sentence)
        user_content = []
        if round_num == 0:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})

        answer_text = random.choice(ANSWER_PARTIAL).replace("[class_name]", sentence)
        answer_with_seg = f"{answer_text} <|seg|>"
        assistant_content = [{"type": "text", "text": answer_with_seg}]
        messages.append({"role": "assistant", "content": assistant_content})

    item["masks"] = mask_paths
    item["messages"] = messages
    return item


# --- Main entry point ---
if __name__ == "__main__":
    # We still create one instance in the main process to get the image list
    main_refer_api = REFER(data_path, ds, splitBy)
    ref_ids_train = main_refer_api.getRefIds(split="train")
    images_ids_train = main_refer_api.getImgIds(ref_ids=ref_ids_train)
    loaded_images = main_refer_api.loadImgs(image_ids=images_ids_train)
    loaded_images_doubled = loaded_images * 2

    num_processes = 32
    print(f"Starting pool with {num_processes} processes...")

    # --- Step 2: Modify processing loop to use tqdm and imap_unordered ---
    Content = []
    with mp.Pool(processes=num_processes,
                 initializer=init_worker,
                 initargs=(data_path, ds, splitBy)) as pool:

        # Use pool.imap_unordered, which returns an iterator
        # tqdm automatically fetches results from the iterator and updates the progress bar
        # total=len(...) tells tqdm the total number of tasks
        # desc="..." is the description text for the progress bar
        pbar = tqdm(pool.imap_unordered(process_image, loaded_images_doubled),
                    total=len(loaded_images_doubled),
                    desc=f"Processing {ds} dataset")

        # Loop through the iterator, adding each completed task to the results list
        for result in pbar:
            if result is not None:  # Also filter out invalid results here
                Content.append(result)

    # --- Step 3: Remove old code for collecting results ---
    # The 'results' variable no longer exists, the Content list is built in the loop
    # Content = [item for item in results if item is not None]  <-- This line needs to be deleted or replaced

    print(f"\nTotal processed and generated items: {len(Content)}")

    output_filename = f"/efficient_sag4text/playground/data/json_files/{ds}_formatted_two_round_mp.json"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(Content, f, indent=4)

    print(f"JSON file saved to: {output_filename}")