import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import DataLoader
from utils.datasets import *
from transformers import BertTokenizer

import nltk
from tqdm.notebook import tqdm
from models.bert import BertForEmbeddings


device = 'cuda' if torch.cuda.is_available() else 'cpu'
pad_symbol = '<pad>'


def get_embedding_weights(X, path, emb_size):
    sentence_tokens = [nltk.word_tokenize(sentence) for sentence in X]

    dictionary = {word for tokens in sentence_tokens for word in tokens}
    dictionary.add(pad_symbol)
    word2idx = {word: index for index, word in enumerate(dictionary)}

    matrix_len = len(dictionary)
    weights_matrix = np.zeros((matrix_len, emb_size))

    embeddings_dict = dict()
    with open(path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            word, vec = line.split(' ', 1)
            if word in dictionary:
                embeddings_dict[word] = np.fromstring(vec, sep=' ')

    for word, index in tqdm(word2idx.items()):
        try:
            weights_matrix[index] = embeddings_dict[word]
        except KeyError:
            weights_matrix[index] = np.random.uniform(size=emb_size)

    return weights_matrix, word2idx


def sentences_to_idx(X, word2idx, max_len):
    sentence_tokens = [nltk.word_tokenize(sentence) for sentence in X]
    for i, sentence_token in enumerate(sentence_tokens):
        tokens_to_add = max(0, max_len - len(sentence_token))
        sentence_tokens[i] += ([pad_symbol] * tokens_to_add)[:max_len]

    return np.array([[word2idx[word] for word in sentence] for sentence in sentence_tokens])


def get_dataloaders(X, y, test_size=0.2, batch_size=32):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    train_dataset = EmotionsDataset(X_train, y_train)
    val_dataset = EmotionsDataset(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader


def preprocess_data(X, y, path, emb_size, max_len):
    embedding_weights, word2idx = get_embedding_weights(X, path, emb_size)
    label_encoder = LabelEncoder().fit(y)

    X = sentences_to_idx(X, word2idx, max_len)
    y = label_encoder.transform(y)

    return X, y, word2idx, embedding_weights


def save_embedding_layer(classification_model, params, path_to_load, path_to_save):
    model = classification_model(*params).to(device)
    model.load_state_dict(torch.load(path_to_load))
    model.eval()

    modules = [module for module in model.modules()]
    torch.save(modules[1].state_dict(), path_to_save)
    print('Saved!')


def get_embeddings(X, embedding_model, params, path):
    model = embedding_model(*params).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    sentence_embeddings = []
    for sentence in tqdm(X):
        sentence_embeddings.append(
            model(torch.tensor(sentence).long().unsqueeze(0).to(device)).squeeze(0).detach().cpu().numpy())

    sentence_embeddings = np.array(sentence_embeddings)
    return sentence_embeddings


def save_embeddings(embeddings, path):
    columns = ['emb_dim_{}'.format(i + 1) for i in range(embeddings.shape[1])]

    pd.DataFrame(embeddings, columns=columns).to_csv(path, index=False)
    print('Saved!')


def get_bert_dataloaders(X, y, test_size=0.2, batch_size=16):
    label_encoder = LabelEncoder().fit(y)

    X, y = X.values, label_encoder.transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=42, stratify=y)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_train_text = tokenizer(list(X_train), truncation=True, padding=True)
    tokenized_val_text = tokenizer(list(X_val), truncation=True, padding=True)

    train_dataset = BertDataset(tokenized_train_text, y_train)
    val_dataset = BertDataset(tokenized_val_text, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader


def get_bert_embeddings(X, path):
    model = BertForEmbeddings().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_text = tokenizer(list(X), truncation=True, padding=True)

    sentence_embeddings = []
    for input_id, attention_mask in tqdm(zip(tokenized_text['input_ids'], tokenized_text['attention_mask'])):
        input_id, attention_mask = torch.tensor(input_id).to(device), torch.tensor(attention_mask).to(device)

        sentence_embeddings.append(
            model(input_id.unsqueeze(0), attention_mask.unsqueeze(0)).squeeze(0).detach().cpu().numpy())

    sentence_embeddings = np.array(sentence_embeddings)
    return sentence_embeddings


def get_embeddings_dataloaders(path, y, batch_size=32):
    X = pd.read_csv(path).values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
