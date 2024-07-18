from torch import nn


class TokenEmbedding(nn.Embedding):

    def __init__(self, vocab_size, embed_size=512):
        super(TokenEmbedding, self).__init__(
            vocab_size, embed_size, padding_idx=1)
