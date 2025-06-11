import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def generate_embeddings_bert(texts, model_name="bert-base-uncased", batch_size=8, max_length=256):
    """
    Generate [CLS]-based embeddings for a list of texts using a pretrained BERT-like model.

    Args:
        texts (List[str]): List of raw input strings.
        model_name (str): Hugging Face model name.
        batch_size (int): Batch size for inference.
        max_length (int): Max token length.

    Returns:
        torch.Tensor: Tensor of shape (N, hidden_size) with embeddings.
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    device = "mps" if torch.backends.mps.is_built() else "cpu"
    model.to(device)

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim=768) using [cls] token embeddings because of text classification
            all_embeddings.append(cls_embeddings.cpu())

        # with torch.no_grad():
        #     outputs = model(**tokens)
        #
        # # ----------- 🔄 Mean Pooling Function --------------
        # def mean_pooling(model_output, attention_mask):
        #     token_embeddings = model_output.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)
        #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #     pooled = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        #     pooled /= input_mask_expanded.sum(dim=1)  # avoid dividing by zero
        #     return pooled  # shape: (batch_size, hidden_dim)
        #
        # # ----------------------------------------------------
        #
        # # Apply mean pooling to get sentence embeddings
        # mean_embeddings = mean_pooling(outputs, tokens['attention_mask'])

    return torch.cat(all_embeddings, dim=0)

df = pd.read_csv("/Users/komalkrishnamogilipalepu/Downloads/archive/train_dataset.csv")
texts = df['text'].tolist()

embeddings = generate_embeddings_bert(texts)

# Optional: Save to disk
torch.save(embeddings, "/Users/komalkrishnamogilipalepu/Downloads/archive/bert_train_embeddings.pt")
