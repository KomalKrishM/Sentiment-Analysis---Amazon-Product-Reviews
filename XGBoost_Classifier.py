import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load training and test data from .pt files
x_train = torch.load("/Users/komalkrishnamogilipalepu/Downloads/archive/bert_train_embeddings.pt")  # should be a tuple (X_train, y_train)
x_test = torch.load("/Users/komalkrishnamogilipalepu/Downloads/archive/bert_test_embeddings.pt")    # should be a tuple (X_test, y_test)

df = pd.read_csv("/Users/komalkrishnamogilipalepu/Downloads/archive/train_dataset.csv")
y_train = df['label'].tolist()

df = pd.read_csv("/Users/komalkrishnamogilipalepu/Downloads/archive/test_dataset.csv")
y_test = df['label'].tolist()

# Train an XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(x_train, y_train)

model.save_model("xgb_review_classifier.json")

# Predict
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1] if len(set(y_train)) == 2 else None

# Evaluate
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average='weighted'),
    "Recall": recall_score(y_test, y_pred, average='weighted'),
    "F1 Score": f1_score(y_test, y_pred, average='weighted'),
}

if y_proba is not None:
    metrics["ROC AUC"] = roc_auc_score(y_test, y_proba)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Metrics:")
for key, val in metrics.items():
    print(f"{key}: {val:.4f}")
