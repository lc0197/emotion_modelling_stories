import torch
import torch.nn.functional as F
from . import device
import torch.nn as nn

import numpy as np

from .common import EmbeddingLayer

from .rnn import LSTMLayer


def create_2d_mask(length, max_length):
    """
    Auxiliary mask to create a 2D mask
    @param length: length of the sequence to consider
    @param max_length: length of the sequence
    @return: 2D mask tensor
    """
    m = torch.zeros((length, length))
    m = F.pad(m, pad=(0, max_length - length, 0, max_length - length), value=1.)
    m = m - torch.eye(max_length)
    m = torch.clip(m, min=0)
    return m.bool()


def get_3d_masks(lengths, max_length, num_heads):
    lengths = [int(l) for l in lengths.detach().cpu().numpy()]
    masks = []
    for l in lengths:
        masks.extend([create_2d_mask(l, max_length) for _ in range(num_heads)])
    return torch.stack(masks)


class TransformerLayer(nn.Module):
    """
    Transformer encoder
    """
    def __init__(self, params):
        super().__init__()

        self.d_model = params.emb_input_dim
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=params.num_heads,
                                                            batch_first=True, dropout=params.transformer_dropout,
                                                            dim_feedforward=params.dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer,
                                                         num_layers=params.num_transformer_layers)
        self.num_heads = params.num_heads
        self.window_mask = params.mask_windowing

    def forward(self, inp):
        # construct attention masks
        encodings, masks, lengths, extras = inp

        if self.window_mask > 0:
            attn_mask = get_3d_masks(lengths, encodings.shape[1], self.num_heads).to(device)
            attn_mask = window_mask(attn_mask, self.window_mask)
            encodings = self.transformer_encoder(encodings, mask=attn_mask)
        else:
            # only padding relevant
            key_mask = torch.where(masks > 0, 0, 1).bool().to(device)
            encodings = self.transformer_encoder(encodings, src_key_padding_mask=key_mask)

        return encodings, masks, lengths, extras


class LSTMTransformerRegressor(nn.Module):
    """
    Embedding layer + LSTM + Transformer + Dropout + Linear + Sigmoid
    """

    def __init__(self, params):
        super().__init__()
        self.embeddings = EmbeddingLayer(params)
        params.emb_input_dim = self.embeddings.embedding_dim
        self.lstm = LSTMLayer(params)
        self.lstm_dropout = nn.Dropout(0.2)
        params.emb_input_dim = 2 * params.lstm_out_dim
        self.transformer = TransformerLayer(params)
        self.dropout = nn.Dropout(params.dropout)
        self.out = nn.Linear(self.transformer.d_model, len(params.label_cols))
        self.activation = nn.Sigmoid()

    def forward(self, inp):
        embedded = self.embeddings(inp)
        lstm_encs, masks, lengths, extras = self.lstm(embedded)
        lstm_encs = self.lstm_dropout(lstm_encs)
        transformer_encs, _, _, _ = self.transformer((lstm_encs, masks, lengths, extras))
        transformer_encs = self.dropout(transformer_encs)

        logits = self.out(transformer_encs)
        return self.activation(logits)


def window_mask(mask, neighbors=3):
    m = np.zeros((mask.shape[1], mask.shape[2]))
    for i in range(mask.shape[1]):
        m[i][max(0, i - neighbors):i + neighbors + 1] = 1
    # false -> attend, true -> do not attend
    m = np.invert(m.astype(bool))
    return torch.logical_or(torch.Tensor(m).to(device), mask)
