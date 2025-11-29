import os
import numpy as np
from PIL import Image
import random
from pycocotools import mask
from pycocotools.coco import COCO
from refer import REFER
import matplotlib.pyplot as plt
# Assuming these lists are defined in a local file as in your setup
from question_answer_list import QUESTION_ALL, QUESTION_PARTIAL, QUESTION_CONDITION, ANSWER_ALL, ANSWER_PARTIAL, \
    ANSWER_CONDITION
# Assuming this utility function is also defined locally
# from utils import encode_mask
import json
import uuid
import multiprocessing as mp
from tqdm import tqdm

# --- Settings ---
# It's recommended to use absolute paths to avoid ambiguity
data_path = "/apdcephfs_qy4/share_302593112/realzliu/dataset_open/lisa_dataset/refer_seg"
# Ensure you set the desired dataset, e.g., "refcoco", "refcoco+", "refcocog", "refclef"
ds = "refclef"
mask_save_dir = f"/efficient_sag4text/playground/data/masks_new/{ds}/"
os.makedirs(mask_save_dir, exist_ok=True)
if ds == "refcocog":
    splitBy = "umd"
else:
    splitBy = "unc"

# --- Initializer for worker processes ---
# This global variable will hold the REFER object for each worker process
refer_api = None


def init_worker(d_path, d_set, s_by):
    """
    Initializes the expensive REFER object once per worker process.
    """
    global refer_api
    refer_api = REFER(d_path, d_set, s_by)


# --- The worker function ---
def process_image(image_info):
    """
    Processes a single image. It iterates through ALL objects (refs) and ALL
    their sentences, creating a distinct training item for each sentence.
    It returns a list of these items.
    """
    global refer_api

    refs = refer_api.imgToRefs.get(image_info["id"], [])
    if not refs:
        return []  # Return an empty list if no references are found

    items_for_this_image = []

    if ds == "refclef":
        image_path = os.path.join("refer_seg/images/saiapr_tc-12", image_info["file_name"])
    else:
        image_path = os.path.join("refer_seg/images/coco_2014/train2014", image_info["file_name"])

    # Iterate over all references (objects) in the image
    for ref in refs:
        sentences = ref['sentences']
        ann = refer_api.refToAnn[ref['ref_id']]

        # --- Mask processing (done once per object) ---
        if isinstance(ann["segmentation"][0], list):  # Polygon format
            rle = mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
        else:  # RLE format
            rle = ann["segmentation"]

        m = mask.decode(rle)
        if m.ndim == 3:
            m = np.sum(m, axis=2)
        m[m > 0] = 1  # Ensure binary mask

        mask_gt = Image.fromarray(m.astype(np.uint8) * 255)
        mask_filename = f"{image_info['id']}_{ref['ref_id']}_{uuid.uuid4()}.png"
        mask_path = os.path.join(mask_save_dir, mask_filename)
        mask_gt.save(mask_path)

        # Iterate over all sentences for the current object
        for sentence_info in sentences:
            item = {}
            messages = []

            sentence = sentence_info['sent']

            # Construct the conversation for this specific sentence
            question = random.choice(QUESTION_PARTIAL).replace("[class_name]", sentence)
            user_content = [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
            messages.append({"role": "user", "content": user_content})

            answer_text = random.choice(ANSWER_PARTIAL).replace("[class_name]", sentence)
            answer_with_seg = f"{answer_text} <|seg|>"
            assistant_content = [{"type": "text", "text": answer_with_seg}]
            messages.append({"role": "assistant", "content": assistant_content})

            # Populate the final item dictionary
            item["images"] = [image_path]
            item["masks"] = [mask_path]  # Each item links to the one mask for this object
            item["messages"] = messages

            items_for_this_image.append(item)

    return items_for_this_image


# --- Main entry point ---
if __name__ == "__main__":
    # Create one instance in the main process to get the image list
    main_refer_api = REFER(data_path, ds, splitBy)
    ref_ids_train = main_refer_api.getRefIds(split="train")
    images_ids_train = main_refer_api.getImgIds(ref_ids=ref_ids_train)
    loaded_images = main_refer_api.loadImgs(image_ids=images_ids_train)

    # --- KEY CHANGE RESTORED ---
    # To replicate the `for _ in range(2):` loop from the original script,
    # we simply process the entire list of images twice. This effectively
    # doubles the dataset, creating different question/answer pairs
    # on the second pass due to the random.choice calls in process_image.
    loaded_images_doubled = loaded_images * 2

    num_processes = 32
    print(f"Starting pool with {num_processes} processes...")

    Content = []
    with mp.Pool(processes=num_processes,
                 initializer=init_worker,
                 initargs=(data_path, ds, splitBy)) as pool:

        # Use the doubled list for processing. Tqdm provides the progress bar.
        pbar = tqdm(pool.imap_unordered(process_image, loaded_images_doubled),
                    total=len(loaded_images_doubled),
                    desc=f"Processing {ds} dataset")

        # Each 'result' from the pool is a LIST of items for one image.
        for result in pbar:
            if result:  # Check if the list is not empty
                Content.extend(result)  # Use extend to add all items from the list

    print(f"\nTotal processed and generated items: {len(Content)}")

    output_filename = f"/efficient_sag4text/playground/data/json_files/{ds}_formatted_all_sentences_doubled_mp.json"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(Content, f, indent=4)

    print(f"JSON file saved to: {output_filename}")