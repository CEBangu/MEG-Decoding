import argparse
import wandb
import torch


## custom imports 

from datahandling import ViTDataHandler
from models.MEGViTNet import MEGVisionTransformer
from experiment import vit_sweep_kfold

def main():

    parser = argparse.ArgumentParser(description="This script runs the KFold Validation parameter sweep for the vision transfromer model(s)")

    parser.add_argument('--model_path', type=str, required=True, help="The HF Model Path")
    parser.add_argument('--sweep_name', type=str, required=True, help="Name of the sweep")
    parser.add_argument('--freeze_type', type=str, required=True, help="The kind of layer freezing you want to apply")
    parser.add_argument('--dataset', type=str, required=True, help="the dataset you want to train on")
    parser.add_argument('--wandb_key', type=str, required=True, help="Wandb API key")


    args = parser.parse_args()

    data_dict = {
        "all_scalograms": ["labels_path", "data_path"], # need to find out what these actually are
        "averaged_scalograms": ["labels_path", "data_path"]
    }

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


    wandb_key = args.wandb_key
    wandb.login(key=wandb_key)
    sweep_name = args.sweep_name
    sweep_id = wandb.sweep(sweep_config, project=sweep_name)

    data = args.dataset
    data_handler = ViTDataHandler(label_path=data_dict[data][0], 
                             image_path=data_dict[data][1],
                             processor_path=args.model_path)

    dataset = data_handler.dataset
    processor = data_handler.processor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_type = args.freeze_type

    wandb.agent(sweep_id, 
                function=lambda:vit_sweep_kfold(
                    train_dataset=dataset,
                    train_dataset_processor=processor,
                    model_class=MEGVisionTransformer,
                    model_name=args.model_path,
                    num_classes=3,
                    config=sweep_config,
                    freeze_type=args.freeze_type,
                    k=10
                ),
                count=10
                )
    
if __name__ == "__main__":
    main()

    