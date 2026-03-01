import torch
import pandas as pd
import numpy as np
import re
import emoji
from sklearn.metrics import f1_score, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = emoji.demojize(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_on_test():
    # load test inputs and labels
    test_df = pd.read_csv("data/test.csv")
    labels_df = pd.read_csv("data/test_labels.csv")

    # join by id
    df = test_df.merge(labels_df, on="id")
    df = df[["comment_text"] + LABELS]
    df["comment_text"] = df["comment_text"].apply(clean_text)

    # filter out examples with any unlabeled (-1) fields
    initial = len(df)
    mask = ~(df[LABELS] == -1).any(axis=1)
    df = df[mask].reset_index(drop=True)

    texts = df["comment_text"].values
    true_labels = df[LABELS].astype(int).values

    print(f"Loaded {initial} test examples; evaluating on {len(texts)} fully-labelled examples")

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained("models/final_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/final_model")
    model.to(DEVICE)
    model.eval()

    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)

    all_preds = []
    batch_size = 32
    from tqdm import tqdm
    total = len(texts)
    try:
        with torch.no_grad():
            for i in tqdm(range(0, total, batch_size), desc="Inferencing", unit="batch"):
                batch = {k: torch.tensor(v[i:i+batch_size]).to(DEVICE) for k, v in encodings.items()}
                outputs = model(**batch)
                logits = outputs.logits.cpu().numpy()
                preds = (logits > 0).astype(int)
                all_preds.append(preds)
    except KeyboardInterrupt:
        print("Evaluation interrupted by user, proceeding with collected predictions...")
    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
    else:
        all_preds = np.zeros((0, len(LABELS)), dtype=int)

    print("Test classification report (per-label):\n")
    print(classification_report(true_labels, all_preds, target_names=LABELS, zero_division=0))

    f1_micro = f1_score(true_labels, all_preds, average="micro")
    f1_macro = f1_score(true_labels, all_preds, average="macro")
    print(f"F1 micro: {f1_micro:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")


if __name__ == "__main__":
    evaluate_on_test()