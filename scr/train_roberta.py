import zipfile
import os
import torch
import pandas as pd
import numpy as np
import re
import emoji
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from focal_loss import FocalLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Unzip datasets if needed
zip_files = [
    "data/jigsaw-toxic-comment-classification-challenge.zip",
    "data/train.csv.zip",
]
for zip_file in zip_files:
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("data/")
        print(f"Extracted {zip_file}")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = emoji.demojize(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train_roberta():
    print("Loading data...")
    df = pd.read_csv("data/train.csv")
    df = df[["comment_text"] + LABELS].dropna()
    df["comment_text"] = df["comment_text"].apply(clean_text)
    
    # Use full dataset for RoBERTa (better model can handle more data)
    df_train = df.sample(n=min(50000, len(df)), random_state=42)
    print(f"Training on {len(df_train)} examples")
    
    # Calculate pos_weight for focal loss
    label_counts = df_train[LABELS].sum()
    total = len(df_train)
    pos_weight = torch.tensor(((total - label_counts) / label_counts).values, dtype=torch.float).to(DEVICE)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_train["comment_text"].values,
        df_train[LABELS].values,
        test_size=0.1,
        random_state=42
    )
    
    print("Loading RoBERTa tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    train_encodings = tokenizer(
        list(train_texts), truncation=True, padding=True, max_length=128
    )
    val_encodings = tokenizer(
        list(val_texts), truncation=True, padding=True, max_length=128
    )
    
    class ToxicDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = ToxicDataset(train_encodings, train_labels)
    val_dataset = ToxicDataset(val_encodings, val_labels)
    
    print("Loading RoBERTa model...")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=6,
        problem_type="multi_label_classification"
    )
    model.gradient_checkpointing_enable()
    
    training_args = TrainingArguments(
        output_dir="models/checkpoints_roberta",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        logging_steps=50
    )
    
    # Trainer with focal loss
    class FocalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    trainer = FocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Starting training with RoBERTa + Focal Loss...")
    trainer.train()
    trainer.save_model("models/roberta_focal_model")
    tokenizer.save_pretrained("models/roberta_focal_model")
    print("RoBERTa model saved to models/roberta_focal_model")


if __name__ == "__main__":
    train_roberta()
