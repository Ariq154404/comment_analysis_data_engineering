from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

class BERTSemanticHashing(nn.Module):
    def __init__(self, pretrained_model_name, hash_size):
        super(BERTSemanticHashing, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.hash_layer = nn.Linear(self.bert.config.hidden_size, hash_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation
        hash_codes = self.sigmoid(self.hash_layer(pooled_output))
        return hash_codes

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERTSemanticHashing('bert-base-uncased', hash_size=128)

# Example input
text = "Example text for semantic hashing"
inputs = tokenizer(text, return_tensors='pt')
hash_codes = model(inputs['input_ids'], inputs['attention_mask'])
print(hash_codes)