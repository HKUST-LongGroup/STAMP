import os
import random
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch

from eval.val_utils import run_in_process_evaluation
from data.question_answer_list import QUESTION_PARTIAL  


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
    return list(images), list(masks), list(image_names), list(questions), list(image_paths)


class InProcessEvaluationCallback(TrainerCallback):


    def __init__(self, eval_dataloaders, sam_predictor=None):

        self.eval_dataloaders = eval_dataloaders
        self.sam_predictor = sam_predictor
        self.trainer = None

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = self.trainer.model
        accelerator = self.trainer.accelerator

        if accelerator.is_main_process:
            print(f"\n{'=' * 30}\n[Callback] Triggering in-process evaluation (Step {state.global_step})\n{'=' * 30}\n")
    
        # 1. Switch to evaluation mode (disable Dropout, etc.)
        model.eval()
        if self.sam_predictor:
            self.sam_predictor.model.eval()

        # 2. Use torch.inference_mode() for optimal inference performance
        with torch.inference_mode():
            for split, dataloader in self.eval_dataloaders.items():
                if accelerator.is_main_process:
                    print(f"\n--- Evaluating dataset: {split} ---")

                # 3. Call the core evaluation function
                metrics = run_in_process_evaluation(
                    model=model,
                    accelerator=accelerator,
                    eval_dataloader=dataloader,
                    sam_predictor=self.sam_predictor
                )

                # 4. Only the main process logs and prints the results.
                if accelerator.is_main_process and metrics:
                    # Construct a dictionary of metrics for logging
                    log_metrics = {f"eval/{split}/{k}": v for k, v in metrics.items()}

                    # Using the trainer's log method ensures compatibility with logging tools like wandb, tensorboard, etc.
                    self.trainer.log(log_metrics)

                    # Print to console
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    print(f"--- Evaluation results ({split}): {metrics_str} ---")

                    # (Optional) Write to a local file
                    with open("output_metrics_in_process.txt", "a", encoding="utf-8") as f:
                        f.write(f"Step {state.global_step}, Split {split}: {metrics_str}\n")

        if accelerator.is_main_process:
            print(f"\n{'=' * 30}\n[Callback] All evaluations completed.\n{'=' * 30}\n")

        # 5. Critical: Switch the model back to training mode!
        model.train()