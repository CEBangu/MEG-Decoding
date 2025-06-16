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
import torch



from datahandling import AlexNetDataHandler
from models.MEGAlexNets import AlexNetDescend, AlexNetFinalOnly, AlexNetLongDescend, AlexNetSuddenDescend, AlexNetBigHead, AlexNetMediumHead, AlexNetSmallHead

from experiment import cnn_test, get_optimizer

train_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.0)),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
])

device = "cuda"

def train_set_collate_fn(batch):
    """
    This function handles the data augmentation for the train set (because the dataset class is already created, so we have to apply
    these in the dataloader)
    """
    images, labels = zip(*batch)  # Unpack batch
    images = [train_transforms(img) for img in images]  # Apply train transforms
    return torch.stack(images), torch.tensor(labels)





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
    weights = weights.to(dtype=torch.float32)
    
    # 5) move to device if requested
    if device is not None:
        weights = weights.to(device)
    
    return counts.index.tolist(), counts.to_dict(), chance, weights



def main():

    parser = ArgumentParser(description="CNN test-loop (9 seeds)")
    parser.add_argument("--model_type",   required=True)
    parser.add_argument("--num_classes",  type=int, default=3)
    parser.add_argument("--project_name", required=True)
    parser.add_argument("--train_labels", required=True)
    parser.add_argument("--test_labels",  required=True)
    parser.add_argument("--batch_size",   type=int,  required=True)
    parser.add_argument("--learning_rate",type=float,required=True)
    parser.add_argument("--freeze",       required=True)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer",    required=True)
    parser.add_argument("--transforms",   required=True)
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed passed by SLURM array task")

    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    train_set = AlexNetDataHandler(csv_file=args.train_labels)
    test_set  = AlexNetDataHandler(csv_file=args.test_labels)
    collate   = True if args.transforms == "time" else None

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=6, pin_memory=True,
        collate_fn=train_set_collate_fn if collate else None
    )
    test_loader  = DataLoader(test_set,
                              batch_size=args.batch_size,
                              shuffle=False)


    _, _, _, class_weights = get_label_stats(train_set.data, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.10)


    run = wandb.init(
        project=args.project_name,
        name=f"{args.project_name}_seed{args.seed}",   # readable but optional
        dir=os.getenv("WANDB_DIR"),
    )
    wandb.config.update(vars(args))


    model_cls = {
        "AlexNetFinalOnly":  AlexNetFinalOnly,
        "AlexNetDescend":    AlexNetDescend,
        "AlexNetLongDescend":AlexNetLongDescend,
        "AlexNetSuddenDescend":AlexNetSuddenDescend,
        "AlexNetBigHead":    AlexNetBigHead,
        "AlexNetMediumHead": AlexNetMediumHead,
        "AlexNetSmallHead":  AlexNetSmallHead,
    }[args.model_type]

    model = model_cls(num_classes=args.num_classes).to(device)
    model.freeze_type(freeze_type=args.freeze)

    if args.freeze == "feature":
        param_groups = {"params": model.classifier.parameters(),
                        "lr": args.learning_rate * 20}
    else:
        param_groups = [
            {"params": model.features.parameters(), "lr": args.learning_rate},
            {"params": model.classifier.parameters(), "lr": args.learning_rate * 20}
        ]

    optimizer = get_optimizer(args.optimizer, param_groups,
                              weight_decay=args.weight_decay)


    model, metrics = cnn_test(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_classes=args.num_classes,
        num_epochs=60,
        project_name=args.project_name,   # will be used in timestamped pickle
        device=device,
        save_results=True
    )

    print(f"[seed {args.seed}] accuracy = {metrics['accuracy']:.3f}")
    wandb.finish()

if __name__ == "__main__":
    main()