import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embd_size, pre_embd=None, is_train_embd=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embd_size)
        if pre_embd is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(pre_embd), requires_grad=is_train_embd)

    def forward(self, x):
        return F.relu(self.embedding(x))


class SelfAttentiveNet(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size=300, mlp_hidden=350, r=30, pre_embd=None, is_train_embd=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        if pre_embd is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(pre_embd), requires_grad=is_train_embd)

        initrange = 0.1
        self.encoder = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.Ws1 = nn.Parameter(torch.Tensor(1, mlp_hidden, 2 * hidden_size).uniform_(-initrange, initrange))
        self.Ws2 = nn.Parameter(torch.Tensor(1, r, mlp_hidden).uniform_(-initrange, initrange))

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bs = x.size(0)
        n = x.size(1)

        x = self.embedding(x).float()
        H, _ = self.encoder(x)
        H_T = H.permute(0, 2, 1).contiguous()
        H_T = self.dropout(H_T)

        A = self.tanh(torch.bmm(self.Ws1.repeat(bs, 1, 1), H_T))
        A = torch.bmm(self.Ws2.repeat(bs, 1, 1), A)
        A = self.softmax(A.view(-1, n)).view(bs, -1, n)

        M = torch.bmm(A, H)

        return M.mean(axis=1)

    def encode(self, sentence):
        sentence_tokens = nltk.word_tokenize(sentence)
        x = torch.tensor([word2idx[word] for word in sentence_tokens]).long().unsqueeze(0).to(device)
        return self.forward(x).squeeze(0).detach().cpu().numpy()


class SelfAttentiveClassifierNet(nn.Module):
    def __init__(
            self,
            vocab_size,
            emb_size,
            num_classes,
            hidden_size=300,
            mlp_hidden=350,
            r=30,
            pre_embd=None,
            is_train_embd=True
    ):
        super().__init__()

        self.sentence_emb = SelfAttentiveNet(vocab_size, emb_size, hidden_size, mlp_hidden, r, pre_embd, is_train_embd)
        self.dropout_1 = nn.Dropout(0.5)
        self.linear_1 = nn.Linear(2 * hidden_size, 2000)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(2000, num_classes)

    def forward(self, x):
        bs = x.size(0)

        x = self.sentence_emb(x)
        x = self.linear_1(self.dropout_1(x))
        x = self.relu(x)
        x = self.linear_2(self.dropout_2(x))
        return x
