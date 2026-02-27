import zipfile
import os

# Unzip datasets if not already unzipped
zip_files = [
    "data/jigsaw-toxic-comment-classification-challenge.zip",
    "data/train.csv.zip",
    "data/test.csv.zip",
    "data/test_labels.csv.zip"
]

for zip_file in zip_files:
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("data/")
        print(f"Extracted {zip_file}")

import torch
import pandas as pd
import numpy as np
import re
import emoji

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
