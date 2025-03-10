import argparse
import wandb
import torch
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
    parser.add_argument('--labels', type=str, required=True, help='path to label csv')
    parser.add_argument('--data_dir', type=str, required=True, help="path to image dir")

    args = parser.parse_args()

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "avg_eval_loss", "goal": "minimize"}, # we want to optimize the average eval loss accross folds
        "parameters": {
            "learning_rate": {"values": [1e-5, 3e-5, 5e-5, 1e-4]}, # sweep learning rates (we'll see how many we can do)
            "lr_scheduler_type": {"values": ["linear", "cosine", "constant"]},
            "optim": {"values": ["adamw_torch", "adamw_hf", "adafactor"]}, # have to consider this some more
            "gradient_accumulation_steps": {"values": [1, 4, 8]}, # does this really matter?
            },
            "early_terminate": {
                "type": "hyperband", # stop runs early
                "min_iter": 15,
            }
    }


    wandb.login() # key stored as env var
    sweep_id = wandb.sweep(sweep_config, project=f"VIT_KFold_HyperSweep")

    hf_token = os.get_env("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("login succesful")
    else:
        print("not logged in to HF!")

    data_handler = ViTDataHandler(label_path=args.labels, 
                             image_path=args.data_dir,
                             processor_path=args.model_path)

    dataset = data_handler.dataset
    processor = data_handler.processor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_type = args.freeze_type

    k = args.num_folds

    wandb.agent(sweep_id, 
                function=lambda:vit_sweep_kfold(
                    train_dataset=dataset,
                    train_dataset_processor=processor,
                    model_class=MEGVisionTransformer,
                    model_name=args.model_path,
                    num_classes=3,
                    config=sweep_config,
                    freeze_type=args.freeze_type,
                    k=k
                ),
                count=10
                )
    
if __name__ == "__main__":
    main()

    