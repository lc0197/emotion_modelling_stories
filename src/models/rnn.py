import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import device
from .common import EmbeddingLayer


class LSTMLayer(nn.Module):
    """
    Wraps LSTM layer(s)
    """

    def __init__(self, params):
        super().__init__()
        self.input_dim = params.emb_input_dim
        self.output_dim = params.lstm_out_dim
        self.num_layers = params.lstm_num_layers

        self.rnn = nn.LSTM(input_size=self.input_dim,
                           hidden_size=self.output_dim,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=True).to(device)

    def forward(self, inp):
        encodings, masks, lengths, extras = inp
        seq_length = encodings.shape[1]

        packed_in = pack_padded_sequence(encodings, lengths.data.tolist(), batch_first=True, enforce_sorted=False)
        packed_in = packed_in.to(device)

        packed_out, states = self.rnn(packed_in)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, padding_value=0., total_length=seq_length)

        return output, masks, lengths, extras


class LSTMRegressor(nn.Module):
    """
    Embedding layer + LSTM layer(s) + linear layer + sigmoid
    """
    def __init__(self, params):
        super().__init__()
        self.embeddings = EmbeddingLayer(params)
        params.emb_input_dim = self.embeddings.embedding_dim
        self.rnn = LSTMLayer(params)
        self.dropout = nn.Dropout(params.dropout)
        self.out = nn.Linear(2 * params.lstm_out_dim, len(params.label_cols))
        self.activation = nn.Sigmoid()

    def forward(self, inp):
        embedded = self.embeddings(inp)
        lstm_encodings, masks, lengths, extras = self.rnn(embedded)
        lstm_encodings = self.dropout(lstm_encodings)
        return self.activation(self.out(lstm_encodings))
