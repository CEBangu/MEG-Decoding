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

from experiment import f1_metric, precision_metric, accuracy_metric, recall_metric


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    """Plots the confusion matrix as a heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


def get_optimizer(name, param_groups, weight_decay):
    """Dynamically select an optimizer just like Hugging Face Trainer."""
    Optim = { #get one of these depending on what the run config is
        "adam": torch.optim.Adam,
        "adamw_torch": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    } [name.lower()] 

    if isinstance(param_groups, list):
        # If param_groups is already a list, use it directly
        return Optim(param_groups, weight_decay=weight_decay)
    else:
        # If param_groups is a model, use its parameters
        lr = param_groups.pop("lr")
        return Optim(param_groups["params"], lr=lr, weight_decay=weight_decay)



def cnn_compute_metrics(predictions, labels, num_classes=3):
    """
    This function handles the metric computation for CNN experiments

    """
    predictions = np.asarray(predictions) # enforce types
    labels = np.asarray(labels) # enforce types

    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=predictions, 
                                       references=labels
                                       )
    f1 = f1_metric.compute(predictions=predictions,
                           references=labels,
                           average="weighted",
                           )
    precision = precision_metric.compute(predictions=predictions,
                                         references=labels,
                                         average="weighted",
                                         zero_division=0.0,
                                         )
    recall = recall_metric.compute(predictions=predictions,
                                   references=labels,
                                   average="weighted",
                                   zero_division=0.0,
                                   )
    
    # need to do specificity by hand
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes))) # get confusion matrix for specificity
    
    specificity_per_class = []
    
    for i in range(num_classes): #num classes
        # true negatives
        TN = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i]) 
        FP = np.sum(cm[:, i]) - cm[i, i]

        # compute specificity for each class and avoid dividing by 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0 
        specificity_per_class.append(specificity)
    
    avg_specificity = np.mean(specificity_per_class)

    # plot the confusion matrix - might have to save this for the final training run
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap="Blues", 
                xticklabels=[str(i) for i in range(num_classes)], 
                yticklabels=[str(i) for i in range(num_classes)]
                )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    wandb.log({"confusion_matrix": wandb.Image(fig)}) # log the matrix
    plt.close(fig)

    # log the stats
    wandb.log({ 
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "specificity": avg_specificity,
    })

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "specificity": avg_specificity,
    }
    

def cnn_train_val_wandb(model, train_loader, val_loader, criterion, optimizer, num_classes=3, num_epochs=10, device="mps", fold=0):
    """
    This function handles the train/validation loop for the cnn model(s). 
    It trains the model, keeps tracks of the train and validation results, and logs them to wandb
    Also does early stopping based on validation loss.
    
    """
    best_val_loss = float('inf')
    epochs_no_improvement = 0
    fold_val_losses = []
    # config = wandb.config if config is None else config
    
    ### epoch looping
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], [] # to compute metrics

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=1.0,  # Clip gradients to prevent exploding gradients
                norm_type=2.0  # L2 norm
            )

            optimizer.step()

            running_loss += loss.item() * images.size(0)  # Sum up batch loss
            _, predicted = outputs.max(1) # predicted class
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            ### storing predicitions
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
        
        train_metrics = cnn_compute_metrics(all_preds, all_labels, num_classes=num_classes)

        ### metric storage
        wandb.log({
            "epoch": epoch + 1,
            f"fold_{fold+1}_train_loss": running_loss / len(train_loader.dataset),
            f"fold_{fold+1}_train_accuracy": 100.0 * correct / total,
            **{f"fold_{fold+1}_train_{k}": v for k, v in train_metrics.items()},
        })
        
        ### printing
        print(f"Epoch [{epoch+1}/{num_epochs}] - Fold {fold+1} - Train Loss: {running_loss / len(train_loader.dataset):.4f}, Accuracy: {100.0 * correct / total:.2f}%")
        
        ### Validation 
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        val_preds, val_labels = [], []

        with torch.no_grad():
            for val_images, val_targets in val_loader:
                val_images, val_targets = val_images.to(device), val_targets.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_targets)
                val_loss += loss.item() * val_images.size(0)  # Accumulate batch loss

                _, val_predicted = val_outputs.max(1)
                val_total += val_targets.size(0)
                val_correct += val_predicted.eq(val_targets).sum().item()

                ### storing validation preds and labels
                val_preds.extend(val_predicted.cpu().numpy().tolist())
                val_labels.extend(val_targets.cpu().numpy().tolist())

        ### get metrics
        val_loss /= len(val_loader.dataset)  # Normalize by dataset size
        val_metrics = cnn_compute_metrics(val_preds, val_labels, num_classes=num_classes)
        fold_val_losses.append(val_loss)

        wandb.log({
            "epoch": epoch + 1,
            f"fold_{fold+1}_val_loss": val_loss,
            f"fold_{fold+1}_val_accuracy": 100.0 * val_correct / val_total,
            **{f"fold_{fold+1}_val_{k}": v for k, v in val_metrics.items()},
        })

        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {100.0 * val_correct / val_total:.2f}%")
        
        if len(fold_val_losses) > 10:  # Only start checking for early stopping after 10 epochs
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= 10:  # Early stopping after 10 epochs without improvement
                print("Stopping early - no improvement")
                break  # Early stopping after 10 epochs without improvement

    return np.mean(fold_val_losses)



def cnn_sweep_train(model_type, model_class, device, k, num_classes, dataset, project_name=None):
    """
    This function handles the wandb parameter sweep for a given cnn model architecture

    """
    if project_name is None:
        project_name = f"{model_type}_KFold_HyperSweep"
    else:
        project_name = project_name
    wandb_dir = os.getenv("WANDB_DIR")
    
    run = wandb.init(
        project=project_name,
        dir=wandb_dir
    )

    config = wandb.config
    group_name = f"CNN_lr:{config.learning_rate}_optim:{config.optimizer}_batch:{config.batch_size}_wd:{config.weight_decay}_freeze:{config.freeze_type}_transforms:{config.transforms}"
    run.name = group_name

    labels = dataset.data.iloc[:, 1].values

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_index, val_index) in enumerate(skf.split(np.arange(len(dataset)), labels)):
        print(f"Fold {fold+1}/{k}")

        train_subset = Subset(dataset, train_index.tolist())
        val_subset = Subset(dataset, val_index.tolist())

        # let's see if some transformations help?
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
        
        collate = True if config.transforms == "time_color" else None

        train_loader = DataLoader(
            train_subset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=6, 
            pin_memory=True,
            collate_fn=train_set_collate_fn if collate else None, #transforms only for training set
            )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=config.batch_size, 
            shuffle=False
            )

        # Ensure no overlap between train and validation sets
        assert not set(train_index).intersection(set(val_index)), "Train and validation sets overlap!"

        # Count the number of labels in each set
        train_labels = [dataset.data.iloc[i, 1] for i in train_index]
        val_labels = [dataset.data.iloc[i, 1] for i in val_index]

        unique_train_labels, train_counts = np.unique(np.array(train_labels), return_counts=True)
        unique_val_labels, val_counts = np.unique(np.array(val_labels), return_counts=True)

        train_labels_dict = dict(zip(unique_train_labels, train_counts))
        val_labels_dict = dict(zip(unique_val_labels, val_counts))

        amounts = [amount for _, amount in val_labels_dict.items()]
        chance_level = max(amounts)/sum(amounts) * 100
        print("Train label distribution:", train_labels_dict)
        print("Validation label distribution:", val_labels_dict)
        print("Chance level:", chance_level)

        class_weights = torch.tensor([amount for _, amount in train_labels_dict.items()], dtype=torch.float)
        class_weights = 1.0/class_weights
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)

        model = model_class(num_classes=num_classes).to(device=device)
        # model.reset_parameters()
        model.freeze_type(freeze_type=config.freeze_type)

        # from dash et al
        if config.freeze_type == "feature": 
            pg = {"params": model.classifier.parameters(), "lr": config.learning_rate * 20}
        else: 
            pg = [{"params": model.features.parameters(), "lr": config.learning_rate},
                  {"params": model.classifier.parameters(), "lr": config.learning_rate * 20}]

        optimizer = get_optimizer(
            name=config.optimizer, 
            param_groups=pg, 
            weight_decay=config.weight_decay
        )

        avg_val_loss = cnn_train_val_wandb(model=model, 
                                       train_loader=train_loader, 
                                       val_loader=val_loader, 
                                       criterion=criterion, 
                                       optimizer=optimizer,
                                       num_classes=num_classes, 
                                       num_epochs=60, # for the smaller datasets, 40 is not enough, and 60 seemed too short as well.
                                       device=device,  
                                       fold=fold)
        fold_results.append(avg_val_loss)


        wandb.log({f"fold_{fold+1}_val_loss": avg_val_loss})

        del train_loader, val_loader, model, optimizer


  
    wandb.log({"avg_val_loss": np.mean(fold_results)})

    wandb.finish()    



def cnn_test(model, train_loader, test_loader, optimizer, criterion, num_classes, num_epochs, device='cuda'):
    """
    This funciton handles the testing loop for the cnn model(s).
    It trains the model, keeps tracks of the train and test results, and logs them to wandb
    """
    test_metrics = {}


    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], [] # to compute metrics

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backwards()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=1.0,  # Clip gradients to prevent exploding gradients
                norm_type=2.0  # L2 norm
            )

            optimizer.step()
            
            running_loss += loss.item() * images.size(0)  # Sum up batch loss
            _, predicted = outputs.max(1) # predicted class
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            ### storing predicitions
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        train_metrics = cnn_compute_metrics(all_preds, all_labels, num_classes=num_classes)

        ### metric storage
        wandb.log({
            "epoch": epoch + 1,
            f"train_loss": running_loss / len(train_loader.dataset),
            f"train_accuracy": 100.0 * correct / total,
            **{f"train_{k}": v for k, v in train_metrics.items()},
        })
        
        ### printing
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss / len(train_loader.dataset):.4f}, Accuracy: {100.0 * correct / total:.2f}%")
        
        ### Validation 
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        
        test_preds, test_labels = [], []

        with torch.no_grad():
            for test_images, test_targets in test_loader:
                test_images, test_targets = test_images.to(device), test_targets.to(device)
                test_outputs = model(test_images)
                loss = criterion(test_outputs, test_targets)
                test_loss += loss.item() * test_images.size(0)  # Accumulate batch loss

                _, test_predicted = test_outputs.max(1)
                test_total += test_targets.size(0)
                test_correct += test_predicted.eq(test_targets).sum().item()

                ### storing testidation preds and labels
                test_preds.extend(test_predicted.cpu().numpy().tolist())
                test_labels.extend(test_targets.cpu().numpy().tolist())

        ### get metrics
        test_loss /= len(test_loader.dataset)  # Normalize by dataset size
        test_metrics = cnn_compute_metrics(test_preds, test_labels, num_classes=num_classes)

        wandb.log({
            "epoch": epoch + 1,
            f"test_loss": test_loss,
            f"test_accuracy": 100.0 * test_correct / test_total,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })

        print(f"Validation - Loss: {test_loss:.4f}, Accuracy: {100.0 * test_correct / test_total:.2f}%")

    return model, test_metrics