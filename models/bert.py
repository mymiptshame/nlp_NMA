import torch
import torch.nn as nn
from transformers import BertModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BertForEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask)
        embeddings = x[0][:, 0, :]
        return embeddings


class ClassificationModel(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()

        self.embedding = BertForEmbeddings()
        self.dropout_1 = nn.Dropout(0.5)
        self.linear_1 = nn.Linear(2 * hidden_size, 2000)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(2000, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids, attention_mask)
        x = self.linear_1(self.dropout_1(x))
        x = self.relu(x)
        x = self.linear_2(self.dropout_2(x))
        return x
