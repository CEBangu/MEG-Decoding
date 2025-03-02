import argparse
import os
import wandb
import torch

## custom imports

from datahandling import AlexNetDataClass
from MEGAlexNets import AlexNetMPSDescend, AlexNetMPSFinalOnly, AlexNetMPSLongDescend, AlexNetMPSSuddenDescend
from experiment import sweep_train



def main():
    parser = argparse.ArgumentParser(description="This script runs the KFold validation and Parameter Sweep for the CNN models")

    parser.add_argument('--model_type', type=str, required=True, help="The model architecture you want to train")
    parser.add_argument('--freeze_type', type=str, required=True, help="The kind of layer freezing you want to apply")
    parser.add_argument('--dataset', type=str, required=True, help="the dataset you want to train on. Can be 'all_scalograms' or 'averaged_scalograms'") #update as needed
    parser.add_argument('--wandb_key', type=str, required=True, help="Wandb api key for tracking login") 

    args = parser.parse_args()

    model_dict = {
        "AlexNetFinalOnly": AlexNetMPSFinalOnly,
        "AlexNetSuddenDescend": AlexNetMPSDescend,
        "AlexNetLongDescend" : AlexNetMPSLongDescend,
        "AlexNetSuddenDescend": AlexNetMPSSuddenDescend
    }

    data_dict = {
        "all_scalograms": ["labels_path", "data_path"], # need to find out what these actually are
        "averaged_scalograms": ["labels_path", "data_path"]
    }

    sweep_config = {
    "method": "bayes",
    "metric": {"name": "avg_val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [1e-4, 3e-4, 1e-3]},
        "batch_size": {"values": [16, 32]},
        "optimizer": {"values": ["adam", "sgd"]},
        "weight_decay": {"values": [0.0, 1e-4, 1e-3]} # regularization - do we really need this?
    },
    "early_terminate": { # stop training if its not working. 
        "type": "hyperband",
        "min_iter": 15
    }
}
    wandb_key = args.wandb_key
    wandb.login(key=wandb_key)
    sweep_id = wandb.sweep(sweep_config, project="CNN_KFold_HyperSweep")

    data = args.dataset 
    dataset = AlexNetDataClass(csv_file=data_dict[data][0],
                               img_directory=data_dict[data][1]
                               )

    mode_class = model_dict[args.model_type]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_type = args.freeze_type

    wandb.agent(sweep_id, 
                function=lambda:sweep_train(
                    dataset=dataset, 
                    model_class=mode_class, 
                    device=device,
                    k=10,
                    freeze_type=freeze_type
                ),
                count=10)


if __name__ == "__main__":
    main()