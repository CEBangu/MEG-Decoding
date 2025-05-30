import torch.optim as optim 
import torch
import torch.nn as nn
import wandb
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from collections import Counter
from argparse import ArgumentParser



from datahandling import AlexNetDataHandler
from models.MEGAlexNets import AlexNetDescend, AlexNetFinalOnly, AlexNetLongDescend, AlexNetSuddenDescend, AlexNetBigHead, AlexNetMediumHead, AlexNetSmallHead

from experiment import cnn_test, get_optimizer

train_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.0)),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
])

def train_set_collate_fn(batch):
    """
    This function handles the data augmentation for the train set (because the dataset class is already created, so we have to apply
    these in the dataloader)
    """
    images, labels = zip(*batch)  # Unpack batch
    images = [train_transforms(img) for img in images]  # Apply train transforms
    return torch.stack(images), torch.tensor(labels)

import torch

def get_label_stats(df, device=None):
    counts = df.iloc[:, 1].value_counts().sort_index()
    total = counts.sum()

    #chance
    largest = counts.max()
    chance = largest / total * 100
    
    # 3) print
    print("Label distribution:", counts.to_dict())
    print(f"Chance level: {largest} / {total} = {chance:.1f}%\n")
    
    # 4) build weights on CPU
    freqs = counts.values.astype(float)
    inv = 1.0 / torch.tensor(freqs)
    weights = inv / inv.sum()
    
    # 5) move to device if requested
    if device is not None:
        weights = weights.to(device)
    
    return counts.index.tolist(), counts.to_dict(), chance, weights



def main():

    parser = ArgumentParser(description="This script does the testing for the models")

    parser.add_argument('--model_type', type=str, required=True, help="The model architecture you want to train")
    parser.add_argument('--num_classes', type=int, default=3, help="how many classes")
    parser.add_argument('--project_name', type=str, default=None, help="project name")
    parser.add_argument('--labels', type=str, required=True, help='path to label csv')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size in int')
    parser.add_argument('--learning_rate', type=float, required=True, help='learning rate in float')
    parser.add_argument('--optimizer', type=str, required=True, help='optimizer to use')
    parser.add_argument('--transforms', type=str, require=True, help='type of transforms to use')

    args = parser.parse_args()

    model_dict = {
        "AlexNetFinalOnly": AlexNetFinalOnly,
        "AlexNetDescend": AlexNetDescend,
        "AlexNetLongDescend" : AlexNetLongDescend,
        "AlexNetSuddenDescend": AlexNetSuddenDescend,
        "AlexNetBigHead": AlexNetBigHead,
        "AlexNetMediumHead": AlexNetMediumHead,
        "AlexNetSmallHead": AlexNetSmallHead,
    }    


    ####
    # logging 
    ####
    project_name = args.project_name

    wandb_dir = os.getenv("WANDB_DIR")

    run = wandb.init(project=project_name,
                dir=wandb_dir,
    )

    # train set
    train_set = AlexNetDataHandler(csv_file=args.train_labels)
    collate = True if args.transforms == "time_color" else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        collate_fn=train_set_collate_fn if collate else None
    )

    #test
    test_set = AlexNetDataHandler(csv_file=args.test_labels)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False
    )

    #dataset stats
    # After you split your DataFrame into df_train, df_val:
    print("Train stats:")
    labels_train, train_counts, chance_train, class_weights = get_label_stats(train_set.data, device)
    print("Test stats:")
    labels_val,   val_counts,   chance_val,   _             = get_label_stats(test_set.data)




    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=0.15)

    # model setup
    model_class=args.model_class
    num_classes=args.num_classes
    
    device = "cuda"
    num_epochs=60
    model_class = model_dict[args.model_type]
    model = model_class(num_classes=num_classes).to(device=device)
    model.freeze_type(freeze_type=args.freeze_type)
    

    if args.freeze_type == "feature": 
        pg = {"params": model.classifier.parameters(), "lr": args.learning_rate * 20}
    else: 
        pg = [{"params": model.features.parameters(), "lr": args.learning_rate},
                {"params": model.classifier.parameters(), "lr": args.learning_rate * 20}]


    optimizer = get_optimizer(
        name=args.optimizer, 
        param_groups=pg, 
        weight_decay=args.weight_decay
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)


    model, metrics = cnn_test(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_classes=num_classes,
        num_epochs=num_epochs,
        device='cuda'
    )


    print(metrics)
    torch.save(model.state_dict(), f"{project_name}.pt")
