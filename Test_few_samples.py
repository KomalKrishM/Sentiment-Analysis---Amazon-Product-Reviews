
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

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
device = "mps" if torch.backends.mps.is_built() else "cpu"
model.to(device)
model.eval()

max_length = 256

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

batch_texts = [text_1, text_2, text_3, text_4, text_5, text_6]
batch_labels = [label_1, label_2, label_3, label_4, label_5, label_6]

batch_preds = []

with torch.no_grad():

    encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)


    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    preds = torch.argmax(logits, dim=1)

    batch_preds.extend(preds.cpu().tolist())


print(batch_preds)
print(batch_labels)
