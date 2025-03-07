import argparse
import os
import wandb
import torch

## custom imports

from datahandling import AlexNetDataHandler
from models.MEGAlexNets import AlexNetMPSDescend, AlexNetMPSFinalOnly, AlexNetMPSLongDescend, AlexNetMPSSuddenDescend
from experiment import cnn_sweep_train



def main():
    parser = argparse.ArgumentParser(description="This script runs the KFold validation and Parameter Sweep for the CNN models")

    parser.add_argument('--model_type', type=str, required=True, help="The model architecture you want to train")
    parser.add_argument('--freeze_type', type=str, required=True, help="The kind of layer freezing you want to apply")
    parser.add_argument('--num_folds', type=int, required=True, help="how many k folds")
    parser.add_argument('--labels', type=str, required=True, help='path to label csv')
    parser.add_argument('--data_dir', type=str, required=True, help="path to image directory you want to train on") #update as needed

    args = parser.parse_args()

    model_dict = {
        "AlexNetFinalOnly": AlexNetMPSFinalOnly,
        "AlexNetDescend": AlexNetMPSDescend,
        "AlexNetLongDescend" : AlexNetMPSLongDescend,
        "AlexNetSuddenDescend": AlexNetMPSSuddenDescend
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
    wandb.login() # login api key stored in env var
    sweep_id = wandb.sweep(sweep_config, project=f"{args.model_type}_KFold_HyperSweep")
    labels_csv = args.labels
    image_directory = args.data_dir
    dataset = AlexNetDataHandler(csv_file=labels_csv,
                               img_directory=image_directory
                               )

    model_class = model_dict[args.model_type]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_type = args.freeze_type

    # change later, just want to test it out for now. 
    k = args.num_folds
    wandb.agent(sweep_id, 
                function=lambda:cnn_sweep_train(
                    model_type=args.model_type,
                    dataset=dataset, 
                    model_class=model_class, 
                    device=device,
                    k=k,
                    freeze_type=freeze_type
                ),
                count=1)


if __name__ == "__main__":
    main()