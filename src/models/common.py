import torch.nn as nn

from . import EMBEDDINGS_DIR
import torch
import numpy as np
import os


class EmbeddingLayer(nn.Module):
    """
    Layer for previously saved sentence embeddings
    """

    def __init__(self, params):
        super().__init__()
        embeddings = np.load(os.path.join(EMBEDDINGS_DIR, params.dataset, params.embeddings, 'embeddings.npy'))

        self.embedding_dim = embeddings.shape[1]

        assert (params.padding_idx > embeddings.shape[0]), "Padding index can not be an existing ID"

        embedding_matrix = np.zeros((params.padding_idx + 1, embeddings.shape[1]))
        # IDs start with 0
        embedding_matrix[1:embeddings.shape[0] + 1, :] = embeddings

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True,
                                                      padding_idx=params.padding_idx)
        self.emb_dropout = nn.Dropout(p=params.emb_dropout)

    def forward(self, inp):
        ids, masks, lengths, extras = inp
        return self.emb_dropout(self.embedding(ids)), masks, lengths, extras
