import argparse
import wandb
import torch
import os
from huggingface_hub import login

## custom imports 

from datahandling import ViTDataHandler
from models.MEGViTNet import MEGVisionTransformer
from experiment import vit_sweep_kfold

def main():

    parser = argparse.ArgumentParser(description="This script runs the KFold Validation parameter sweep for the vision transfromer model(s)")

    parser.add_argument('--model_path', type=str, required=True, help="The HF Model Path")
    parser.add_argument('--freeze_type', type=str, required=True, help="The kind of layer freezing you want to apply")
    parser.add_argument('--num_folds', type=int, required=True, help="How many folds")
    parser.add_argument('--num_classes', type=int, default=3, help="number of classes in task")
    parser.add_argument('--project_name', type=str, default=None, help="name of the project, i.e., datatype")
    parser.add_argument('--labels', type=str, required=True, help='path to label csv')
    parser.add_argument('--data_dir', type=str, required=True, help="path to image dir")

    args = parser.parse_args()

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "eval_loss", "goal": "minimize"}, # we want to optimize the average eval loss accross folds
        "parameters": {
            "learning_rate": {"values": [1e-5, 3e-5, 5e-5, 1e-4]}, # sweep learning rates (we'll see how many we can do)
            "lr_scheduler_type": {"values": ["linear", "cosine", "constant"]},
            "optimizer": {"values": ["adamw_torch", "adamw_hf", "adafactor"]}, # have to consider this some more
            "gradient_accumulation_steps": {"values": [1, 4, 8]}, # does this really matter?
            },
            "early_terminate": {
                "type": "hyperband", # stop runs early
                "min_iter": 5,
            }
    }


    wandb.login() # key stored as env var
    sweep_id = wandb.sweep(sweep_config, project=args.project_name)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("login succesful")
    else:
        print("not logged in to HF!")

    dataset = ViTDataHandler(label_path=args.labels, 
                             image_path=args.data_dir,
                             processor_path=args.model_path)


    processor = dataset.processor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_type = args.freeze_type
    num_classes = args.num_classes
    k = args.num_folds

    wandb.agent(sweep_id, 
                function=lambda:vit_sweep_kfold(
                    train_dataset=dataset,
                    train_dataset_processor=processor,
                    model_class=MEGVisionTransformer,
                    model_name=args.model_path,
                    num_classes=num_classes,
                    config=sweep_config,
                    freeze_type=args.freeze_type,
                    k=k,
                    project_name=args.project_name
                ),
                count=3 # number of hyperparameters to search over
                )
    
if __name__ == "__main__":
    main()

    