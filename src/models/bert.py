from transformers import AutoModel, AutoConfig

import torch.nn as nn


class SimpleBERTLikeClassifier(nn.Module):
    """
    Transformer model + Dropout + linear layer
    """

    def __init__(self, num_classes, backbone_id, return_embedding=False):
        '''
        Huggingface Model + classification head
        @param num_classes: number of classes
        @param backbone_id: model id from huggingface model hub
        @param return_embedding: also return the final [CLS] embedding?
        '''
        super().__init__()

        self.encoder = AutoModel.from_pretrained(backbone_id)

        self.dropout = nn.Dropout(0.5)
        conf = AutoConfig.from_pretrained(backbone_id)
        self.linear = nn.Linear(conf.hidden_size, num_classes)
        self.return_embedding = return_embedding

    def forward(self, input):
        ids, masks, lengths = input
        encoding = self.encoder(ids, attention_mask=masks, output_hidden_states=True)[0][:, 0, :]
        encoding_do = self.dropout(encoding)
        logits = self.linear(encoding_do)
        if self.return_embedding:
            return logits, encoding
        else:
            return logits
