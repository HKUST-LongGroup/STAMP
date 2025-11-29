import os
import numpy as np
from PIL import Image
import random
from pycocotools import mask
from grefer import G_REFER  
from question_answer_list import QUESTION_PARTIAL, ANSWER_PARTIAL
import json
import uuid
import multiprocessing as mp
from tqdm import tqdm


data_path = "/apdcephfs_qy4/share_302593112/realzliu/dataset_open/lisa_dataset/refer_seg"
ds = "grefcoco"  
splitBy = "unc"

mask_save_dir = f"/efficient_sag4text/playground/data/gref_masks_new/{ds}/"
os.makedirs(mask_save_dir, exist_ok=True)

refer_api = None


def init_worker(d_path, d_set, s_by):

    global refer_api
    refer_api = G_REFER(d_path, d_set, s_by)



def process_image(image_info):

    global refer_api

    refs = refer_api.imgToRefs.get(image_info["id"], [])
    if not refs:
        return []  

    items_for_this_image = []

    if ds == "refclef":
        image_path = os.path.join("refer_seg/images/saiapr_tc-12", image_info["file_name"])
    else:
        image_path = os.path.join("refer_seg/images/coco_2014/train2014", image_info["file_name"])

    for ref in refs:
        sentences = ref['sentences']
        anns = refer_api.refToAnn.get(ref['ref_id'])

        if None in anns or "segmentation" not in anns[0]:
            m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
        else:
            m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
            for ann in anns:
                if type(ann["segmentation"]) == list and type(ann["segmentation"][0]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"],
                        image_info["height"],
                        image_info["width"],
                    )
                else:
                    pass
                    # rle = ann["segmentation"]
                    # for i in range(len(rle["counts"])):
                    #     if not isinstance(rle["counts"][i], bytes):
                    #         rle["counts"][i] = rle[i]["counts"][i].encode()
                m = m + np.sum(mask.decode(rle), axis=2)
            m[m > 1] = 1

        mask_gt = Image.fromarray(m.astype(np.uint8) * 255)
        mask_filename = f"{image_info['id']}_{ref['ref_id']}_{uuid.uuid4()}.png"
        mask_path = os.path.join(mask_save_dir, mask_filename)
        mask_gt.save(mask_path)

        for sentence_info in sentences:
            item = {}
            messages = []
            sentence = sentence_info['sent']

            question = random.choice(QUESTION_PARTIAL).replace("[class_name]", sentence)
            user_content = [{"type": "image"}, {"type": "text", "text": question}]
            messages.append({"role": "user", "content": user_content})

            answer_text = random.choice(ANSWER_PARTIAL).replace("[class_name]", sentence)
            answer_with_seg = f"{answer_text} <|seg|>"
            assistant_content = [{"type": "text", "text": answer_with_seg}]
            messages.append({"role": "assistant", "content": assistant_content})

            item["images"] = [image_path]
            item["masks"] = [mask_path]  
            item["messages"] = messages

            items_for_this_image.append(item)

    return items_for_this_image


if __name__ == "__main__":
    main_refer_api = G_REFER(data_path, ds, splitBy)
    ref_ids_train = main_refer_api.getRefIds(split="train")
    images_ids_train = main_refer_api.getImgIds(ref_ids=ref_ids_train)
    loaded_images = main_refer_api.loadImgs(image_ids=images_ids_train)

    loaded_images_doubled = loaded_images * 2

    num_processes = 64
    print(f"Starting process pool with {num_processes} processes...")

    Content = []
    with mp.Pool(processes=num_processes,
                 initializer=init_worker,
                 initargs=(data_path, ds, splitBy)) as pool:

        pbar = tqdm(pool.imap_unordered(process_image, loaded_images_doubled),
                    total=len(loaded_images_doubled),
                    desc=f"Processing {ds} dataset")

        # Collect results from each process
        for result in pbar:
            if result:
                Content.extend(result)

    print(f"\nProcessing complete. Generated {len(Content)} items.")

    # Save the final JSON file
    output_filename = f"./datasets/json_files/{ds}_doubled_mp_fullmask.json"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(Content, f, indent=4)

    print(f"JSON file saved to: {output_filename}")
    print(f"All mask images saved to: {mask_save_dir}")