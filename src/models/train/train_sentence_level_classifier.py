import sqlite3

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
from transformers import BertTokenizer, BertForSequenceClassification, EarlyStoppingCallback, TrainingArguments, Trainer

"""
:description Script which Generates and Trains a BERT Classifier to detect if a page of text is referencing 
             a specific concept

"""

################
# EDIT THESE VALUES TO TRAIN DIFFERENT CLASSIFIERS
################
column_name = 'isAcquisition'
model_path = "../../../models/acquisition_sentence_level_model_weights"
################


dat = sqlite3.connect("../../../data/processed/test_train_database.db")
query = dat.execute("SELECT * From DryCleaned  WHERE " + column_name + " = '0' OR " + column_name + " = '1'")
cols = [column[0] for column in query.description]
dataset = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)

# TODO: Update Stratified Data method
# Do rough simple stratification of the data
positives = dataset.loc[dataset[column_name] == '1']
negatives = dataset.loc[dataset[column_name] == '0']

train_ones = positives.sample(frac=0.7)
test_ones = positives.drop(train_ones.index)

# Will require update with more data
sample_size = 3.5 * positives.size / negatives.size
sample_negatives = negatives.sample(frac=sample_size)

train_zeroes = sample_negatives.sample(frac=0.7)
test_zeroes = sample_negatives.drop(train_zeroes.index)

data = pd.concat([train_ones, train_zeroes], ignore_index=True).sample(frac=1).reset_index(drop=True)
devData = pd.concat([test_ones, test_zeroes], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Define pretrained tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ----- 1. Preprocess data -----#
# Preprocess data
X = list(data["text"])
y = list(pd.to_numeric(data[column_name]))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)


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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
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
model.save_pretrained(model_path)

# ----- 3. Predict -----#
# Load test data

X_test = list(devData["text"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Load trained model
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)
y_act = pd.to_numeric(devData[column_name])
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
