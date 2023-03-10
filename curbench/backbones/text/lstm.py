import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, num_labels, embedding_dim=300, hidden_dim=512, 
                 n_layers=3, use_bidirectional=False, use_dropout=False):
        super().__init__()
        self.num_labels = num_labels
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           batch_first=True, bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0.)
        self.fc = nn.Linear(hidden_dim, self.num_labels)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def forward(self, input_ids, **kwargs):
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.rnn(embedded)
        last_hidden = self.dropout(hidden[-1, :, :])
        logits = self.fc(last_hidden)
        return logits, hidden, cell
