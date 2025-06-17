import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, get_scheduler
from torch.optim import AdamW
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

# 1. Load your data
df = pd.read_csv("/Users/komalkrishnamogilipalepu/Downloads/archive/train_dataset.csv")
texts = df['text'].tolist() #[:200000]
labels = df['label'].tolist() #[:200000]

# 2. Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# # 3. PyTorch Dataset
# class SentimentDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=256):
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
#         item = {key: val[idx] for key, val in self.encodings.items()}
#         item["labels"] = self.labels[idx]
#         return item
#
# # 4. Dataset and DataLoader
# dataset = SentimentDataset(texts, labels, tokenizer)
# Prt
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# 5. Custom BERT Classifier
class CustomBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 2 classes: positive/negative

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] # collects the embeddings of [cls] token and apply a feedforward layer with tanh activation
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


# 6. Model setup
device = "mps" if torch.backends.mps.is_built() else "cpu"
model = CustomBERTClassifier().to(device)

# 7. Optimizer, loss, and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(texts) * 3)  # 3 epochs)

# Directory to save
save_directory = "/Users/komalkrishnamogilipalepu/Downloads/LLM"

# Create the directory if not exists
os.makedirs(save_directory, exist_ok=True)

batch_size = 32
max_length = 256

# 8. Training loop
model.train()
for epoch in range(3):
    print(f"\nEpoch {epoch + 1}")
    total_loss = 0

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        batch_labels = torch.tensor(labels[i:i+batch_size]).to(device) #encodings["labels"].to(device)

    # for batch in tqdm(dataloader):
    #     input_ids = batch["input_ids"].to(device)
    #     attention_mask = batch["attention_mask"].to(device)
    #     batch_labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1} Loss: {total_loss / len(texts):.4f}")

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_directory, "SA-APR_fine-tuned_bert_model_Epoch_%d.bin"%epoch))


# Save tokenizer
# tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to: {save_directory}")

