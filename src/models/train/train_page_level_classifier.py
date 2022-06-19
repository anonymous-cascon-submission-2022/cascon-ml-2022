import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import LongformerTokenizer, LongformerForSequenceClassification, EarlyStoppingCallback, \
    TrainingArguments, Trainer

from src import config

"""
:description Script which Generates and Trains a Longformer Classifier to detect if a page of text is referencing an 
             acquisition or Merger

"""

db_path = f'sqlite:///{Path(Path(__file__).parent.parent.parent, config["paths"]["databases"]["test_train"])}'
dat = sqlite3.connect(db_path)
test_col = list(config['data']['attributes'].keys())[0]
query = dat.execute(
    f"SELECT * FROM {config['data']['tables']['page_level']} WHERE {test_col} IS NOT NULL")
cols = [column[0] for column in query.description]
dataset = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)

# TODO: Update Stratified Data method
# Do rough simple stratification of the data
positives = dataset.loc[dataset['isMergerOrAcquisition'] == '1']
negatives = dataset.loc[dataset['isMergerOrAcquisition'] == '0']

train_ones = positives.sample(frac=0.7)
test_ones = positives.drop(train_ones.index)

train_zeroes = negatives.sample(frac=0.7)
test_zeroes = negatives.drop(train_zeroes.index)

data = pd.concat([train_ones, train_zeroes], ignore_index=True).sample(frac=1).reset_index(drop=True)
devData = pd.concat([test_ones, test_zeroes], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Define pretrained tokenizer and model
model_name = "allenai/longformer-base-4096"
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ----- 1. Preprocess data -----#
# Preprocess data
X = list(data["text"])
y = list(pd.to_numeric(data["isMergerOrAcquisition"]))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=3072)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=3072)


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)


# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    fp16=True,
    num_train_epochs=1,
    seed=0,
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
trainer.train()

# Save Model Weights (should overwrite existing weights)
model.save_pretrained("../../../models/page_level_model_weights")

# ----- 3. Predict -----#
# Load test data

X_test = list(devData["text"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=3072)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Load trained model
model_path = "../../../models/page_level_model_weights"
model = LongformerForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
raw_pred, _, _ = test_trainer.prediction_loop(test_loader, description="prediction")

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)
y_act = pd.to_numeric(devData["isMergerOrAcquisition"])
x_text = devData["text"]

# Create table to show
columns = [pd.Series(x_text.values), pd.Series(y_act.values), pd.Series(y_pred)]
model_pred_results = pd.concat(columns, axis=1, ignore_index=True)
model_pred_results = model_pred_results.rename(
    columns={0: "text", 1: "actual", 2: "prediction"}
)
model_pred_results["sum"] = (
        pd.to_numeric(model_pred_results["actual"]) + pd.to_numeric(model_pred_results["prediction"])
)

# Output Result Metrics
print("FAILED TESTS:")
print(model_pred_results.loc[model_pred_results["sum"] == 1])
print("CONFUSION MATRIX\n")
print(confusion_matrix(list(y_act), list(y_pred)))
print("CLASSIFICATION REPORT\n")
print(classification_report(y_act, y_pred))

complete = 1
