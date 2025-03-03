from datasets import Dataset, Image, ClassLabel
from transformers import AutoImageProcessor
import pandas as pd
import os

class ViTDataHandler:
    def __init__(self, label_path, image_path, processor_path):
        self.processor = AutoImageProcessor.from_pretrained(processor_path, use_fast=True)
        self.processor.size = {"height" : 384, "width" : 384} # reccomended to fine-tune with 384x384

        df = pd.read_csv(label_path)
        df['image'] = df['FileName'].apply(lambda x: os.path.join(image_path, x))
        df["label"] = df["Label"].astype("category")
        df = df.drop(columns=["Label", "FileName"])

        class_labels = ClassLabel(names=list(set(df["label"])))
        dataset = Dataset.from_pandas(df).cast_column("image", Image())
        dataset = dataset.cast_column("label", class_labels)

        self.dataset = dataset.map(self._transform_for_model, batched=True)

    def _transform_for_model(self, example):
        """Transform the dataset into the format the model is expecting"""
        example['pixel_values'] = [image.convert("RGB") for image in example['image']]
        example['pixel_values'] = self.processor(example['pixel_values'], return_tensors='pt')["pixel_values"]
        return example