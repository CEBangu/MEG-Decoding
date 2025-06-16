import argparse
import os
import wandb
import torch

## custom imports

from datahandling import AlexNetDataHandler
from models.MEGAlexNets import AlexNetDescend, AlexNetFinalOnly, AlexNetLongDescend, AlexNetSuddenDescend, AlexNetBigHead, AlexNetMediumHead, AlexNetSmallHead
from models.MEGResNets import ResNet101SmallHead, ResNet101LongHead
from experiment import cnn_sweep_train



def main():
    parser = argparse.ArgumentParser(description="This script runs the KFold validation and Parameter Sweep for the CNN models")

    parser.add_argument('--model_type', type=str, required=True, help="The model architecture you want to train")
    parser.add_argument('--num_folds', type=int, required=True, help="how many k folds")
    parser.add_argument('--num_classes', type=int, default=3, help="how many classes")
    parser.add_argument('--project_name', type=str, default=None, help="project name")
    parser.add_argument('--labels', type=str, required=True, help='path to label csv')

    args = parser.parse_args()

    model_dict = {
        "AlexNetFinalOnly": AlexNetFinalOnly,
        "AlexNetDescend": AlexNetDescend,
        "AlexNetLongDescend" : AlexNetLongDescend,
        "AlexNetSuddenDescend": AlexNetSuddenDescend,
        "AlexNetBigHead": AlexNetBigHead,
        "AlexNetMediumHead": AlexNetMediumHead,
        "AlexNetSmallHead": AlexNetSmallHead,
        "ResNet101SmallHead": ResNet101SmallHead,
        "ResNet101LongHead": ResNet101LongHead,
    }

    sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"}, 
    "parameters": {
        "learning_rate": {"values": [1e-3, 1e-4, 1e-5, 3e-4]}, #0.0001, 3e-4]},
        "batch_size": {"values": [128, 64, 256]}, #128]},
        "optimizer": {"values": ["adamw"]}, #, "sgd" "rmsprop", "adam"]},
        "weight_decay": {"values": [1e-4, 1e-5, 0.0, 1e-3]},
        "freeze_type": {"values": ["none", "feature"]},
        "transforms": {"values": ["none",]} # "none", "time_color", "conv", "all",  
    },
    "early_terminate": { # stop training if its not working. 
        "type": "hyperband",
        "min_iter": 20,
    }
}
    wandb.login() # login api key stored in env var
    sweep_id = wandb.sweep(sweep_config, project=args.project_name)
    labels_csv = args.labels
    dataset = AlexNetDataHandler(csv_file=labels_csv)

    model_class = model_dict[args.model_type]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # change later, just want to test it out for now. 
    k = args.num_folds # if we find one that seems to work then maybe switch to 10, otherwise it just takes too long
    k = 9
    num_classes = args.num_classes
    wandb.agent(sweep_id, 
                function=lambda:cnn_sweep_train(
                    model_type=args.model_type,
                    dataset=dataset, 
                    model_class=model_class, 
                    device=device,
                    k=k,
                    num_classes=num_classes,
                    project_name=args.project_name
                ),
                count=50) # need to change the number of hyperparameters searched over.


if __name__ == "__main__":
    main()