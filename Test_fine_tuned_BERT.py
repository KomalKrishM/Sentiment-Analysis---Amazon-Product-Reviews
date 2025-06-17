# Reload tokenizer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Reload model architecture
class CustomBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:,0,:] #pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

# Load weights
model = CustomBERTClassifier()
model.load_state_dict(torch.load("SA-APR_fine-tuned_bert_model_Epoch_1.bin"))
model.to("mps" if torch.backends.mps.is_built() else "cpu")
model.eval()

# Load test data
test_df = pd.read_csv("/Users/komalkrishnamogilipalepu/Downloads/archive/test_dataset.csv")
texts = test_df['text'].tolist() #[:25000]
labels = test_df['label'].tolist() #[:25000]

# Create test dataset
# class SentimentDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=512):
#         self.encodings = tokenizer(
#             texts,
#             truncation=True,
#             padding="max_length",
#             max_length=max_length,
#             return_tensors="pt"
#         )
#         self.labels = torch.tensor(labels)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         item = {k: v[idx] for k, v in self.encodings.items()}
#         item["labels"] = self.labels[idx]
#         return item

# test_dataset = SentimentDataset(texts, labels, tokenizer)
# test_loader = DataLoader(test_dataset, batch_size=16)

# Run inference
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
all_preds, all_labels = [], []

batch_size = 32
max_length = 256

with torch.no_grad():

    # total_loss = 0

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        batch_labels = torch.tensor(labels[i:i+batch_size]).to(device)

    # for batch in test_loader:
    #     input_ids = batch["input_ids"].to(device)
    #     attention_mask = batch["attention_mask"].to(device)
    #     labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

# Accuracy
# acc = accuracy_score(all_labels, all_preds)
# print(f"Test Accuracy: {acc:.4f}")

# Evaluate
metrics = {
    "Accuracy": accuracy_score(all_labels, all_preds),
    "Precision": precision_score(all_labels, all_preds, average='weighted'),
    "Recall": recall_score(all_labels, all_preds,  average='weighted'),
    "F1 Score": f1_score(all_labels, all_preds,  average='weighted'),
}

# if y_proba is not None:
#     metrics["ROC AUC"] = roc_auc_score(y_test, y_proba)

print("Classification Report:\n", classification_report(all_labels, all_preds))
print("Metrics:")
for key, val in metrics.items():
    print(f"{key}: {val:.4f}")

