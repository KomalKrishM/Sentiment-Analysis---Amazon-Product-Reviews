# from generate_embeddings import generate_embeddings_bert
import torch
import xgboost as xgb
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


model = xgb.XGBClassifier()
model.load_model("xgb_review_classifier.json")

text_1 = "Good product at reasonable price: Though the output of this power supply is lower than the Apple supply, \
            it seems to work fine--and it is very reasonably priced. \
            Too bad that Amazon stopped carrying the supply."
label_1 = 1

text_2 = "So much better than the apple version!: I love it, it's LONG and the connection is secure and \
            it doesn't get hot! It's wonderful!"
label_2 = 1

text_3 = "Cheap, Doesn't charge, dangerous, power cord: This is one of the most poorly designed and implemented chargers around. \
            I own a 15\" Powerbook G4 (which this is listed as being compatible with).The problem is that this isn't actually a \
            65watt power cord... so don't expect to use your computer while it's charging... \
            and don't expect to be able to touch the \"brick\" as it's plugged in, you'll burn yourself. \
            I've returned this item and got a charger from another company for just a few dollars more that works a TON better."
label_3 = 0

text_4 = "Off and on: Its hard to find these types of chargers since Mac switched to the magnetic type. \
            Its hit and miss with the working. My battery indicator switches from charging to not in a blinking fashion."
label_4 = 0

text_5 = "Excellent Item!: I miss the lighted ring telling you that the charger is charging, but for 1/2 of the price of \
          Apple's replacement, I can do without it!The LED iS BRIGHT, and cool bluish color. \
            The cable that breaks on Apple's charger is noticeably thicker on this charger.Get this. You'll not regret it."
label_5 = 1

text_6 = "Great product at a great price: A replacement adapter for my 4-year-old G4 Powerbook was going to cost $80 from Apple. \
        The Macally adapter is less than half the price, and it('s built like a tank -- the wires are thicker, and the plastic is \
        more durable. The only thing you might miss is having the yellow/green charging light...but that')s more of a luxury and \
        not a necessity. Highly recommended."
label_6 = 1


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

# x_test = torch.load("/Users/komalkrishnamogilipalepu/Downloads/archive/bert_test_embeddings.pt")
# embeddings = x_test[:6,:]

texts = [text_1, text_2, text_3, text_4, text_5, text_6]
embeddings = generate_embeddings_bert(texts)

# print(x_test.shape)
print(embeddings.shape)

predictions = model.predict(embeddings)

print(predictions)
