import torch
from torch.utils.data import Dataset


class EmotionsDataset(Dataset):
    def __init__(self, sentences, target):
        self.sentences = sentences
        self.target = target

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]).long(), torch.tensor(self.target[idx]).long()

    def __len__(self):
        return len(self.target)


class BertDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.X.items()}
        item['target'] = torch.tensor(self.y[idx]).long()
        return item

    def __len__(self):
        return len(self.y)


class ClassificationDataset(Dataset):
    def __init__(self, sentences, target):
        self.sentences = sentences
        self.target = target

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]).float(), torch.tensor(self.target[idx]).long()

    def __len__(self):
        return len(self.target)
