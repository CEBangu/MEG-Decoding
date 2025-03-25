from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from transformers import Trainer, TrainingArguments
from torch.utils.data import Subset
from collections import Counter
import numpy as np
import torch
import wandb
import os

# custom imports 
from experiment import accuracy_metric, f1_metric, precision_metric, recall_metric
from models.ViTCallbacks import FoldEvalTrackingCallback, TrainMetricsCallback


def collate_fn(examples):
    """This function gets the samples in the right format for the model"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def vit_compute_metrics(eval_pred, num_classes=3):
    """this functions handles the metric computation for the trainer"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1) #logits to class
    accuracy = accuracy_metric.compute(predictions=predictions, 
                                       references=labels)
    f1 = f1_metric.compute(predictions=predictions, 
                           references=labels, 
                           average='weighted',
                           )
    precision = precision_metric.compute(predictions=predictions, 
                                         references=labels, 
                                         average='weighted',
                                         zero_division=0.0 #control null predicition
                                         )
    recall = recall_metric.compute(predictions=predictions, 
                                   references=labels, 
                                   average='weighted',
                                   zero_division=0.0 #control null predicition
                                   )

    # specificity we will have to compute manuall unfortunately
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes))) # classes remember to change this!
    
    specificity_per_class = []
    
    for i in range(num_classes): #num classes
        # true negatives
        TN = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i]) 
        FP = np.sum(cm[:, i]) - cm[i, i]

        # compute specificity for each class and avoid dividing by 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_per_class.append(specificity)
    
    avg_specificity = np.mean(specificity_per_class)

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

def vit_sweep_kfold(train_dataset, train_dataset_processor, model_class, model_name, num_classes=3, config=None, freeze_type=None, k=10, project_name=None):
    """Train dataset has to be dataset["train"]
    Unlike with the CNN, the trainer handles the training/validation loop here
    """
    wandb_dir = os.getenv("WANDB_DIR")
    hf_output_dir = os.getenv("HF_OUTPUT_DIR")

    #wandb configuration setup. 
    if project_name is None:
        project_name = "Google_ViT-KFold-HyperSweep"
    else:
        project_name = project_name

    run=wandb.init(
        project=project_name,
        dir=wandb_dir,
    )

    config=wandb.config
    group_name = f"ViT_lr:{config.learning_rate}_optim:{config.optimizer}_sched:{config.lr_scheduler_type}"#_grads:{config.gradient_accumulation_steps}"
    # Optionally, set tags or name here:
    run.name = group_name

    # decide on the splits
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"Training Fold {fold + 1}/{k}...")

        # remember to reinstantiate the model!!! and freeze the layers for each fold
        
        # model code
        model = model_class.from_pretrained(model_name, 
                                            ignore_mismatched_sizes=True,
                                            num_labels=num_classes
                                            )
        #trainer handles devices
        #apply the freeze type
        model.freeze_type(freeze_type)

        # splitting into train and val
        train_subset = Subset(train_dataset, train_idx.tolist())
        val_subset = Subset(train_dataset, val_idx.tolist())

        train_labels = [train_dataset[idx]["label"] for idx in train_idx]
        val_labels = [train_dataset[idx]["label"] for idx in val_idx]

        # Count label occurrences
        train_label_counts = Counter(train_labels)
        val_label_counts = Counter(val_labels)

        chance_accuracy = max(val_label_counts.values())/sum(val_label_counts.values()) 

        # Print class distributions
        print("Train Set Label Distribution:")
        for label, count in sorted(train_label_counts.items()):
            print(f"Class {label}: {count} samples")

        print("Validation Set Label Distribution:")
        for label, count in sorted(val_label_counts.items()):
            print(f"Class {label}: {count} samples")

        # Print chance accuracy level
        print(f"Chance Accuracy Level for Fold {fold + 1}: {chance_accuracy:.2%}") 

        training_args = TrainingArguments(
            output_dir=hf_output_dir,
            seed=42,
            remove_unused_columns=False,
            eval_strategy="epoch", #maybe epoch is better?
            save_strategy="epoch",
            fp16=True,
            # max_grad_norm=1.0, #gradient clippping
            learning_rate=config.learning_rate, # take it from the wandb config
            lr_scheduler_type=config.lr_scheduler_type, # take from config
            optim=config.optimizer, # tune optimizer
            # gradient_accumulation_steps=config.gradient_accumulation_steps, #tune gradient accumulation
            per_device_train_batch_size=128, # IMPORTANT
            per_device_eval_batch_size=128, # IMPORTANT
            num_train_epochs=20, # IMPORTANT
            # warmup_ratio=0.001,
            logging_steps=1, # change to more later
            metric_for_best_model='eval_loss',
            report_to="wandb",
            push_to_hub=False,
            logging_dir=hf_output_dir,
        )
        trainer = Trainer(
            model=model, 
            args=training_args,
            tokenizer=train_dataset_processor,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            data_collator=collate_fn,
            compute_metrics=vit_compute_metrics,
            callbacks=[FoldEvalTrackingCallback(fold+1)]
        )
        
        trainer.add_callback(TrainMetricsCallback(trainer=trainer))
        trainer.train()
        eval_results = trainer.evaluate()

        wandb.log({f"fold_{fold+1}_eval_loss": eval_results["eval_loss"]})
        fold_results.append(eval_results["eval_loss"]) # is this really what we want to track?
        
    avg_eval_loss=np.mean(fold_results)
    wandb.log({"avg_eval_loss": avg_eval_loss})
    
    run.finish()
    return avg_eval_loss