from torch import nn
from embedding.token import TokenEmbedding
from embedding.position import PositionEmbedding
from embedding.segment import SegmentEmbedding


class BERTEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.token = TokenEmbedding(
            vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionEmbedding(embed_size=embed_size)
        self.segment = SegmentEmbedding(embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)

        assert embed_size == self.token.embedding_dim, 'WRONG EMBEDDING SIZE'

    def forward(self, sequence, segment_label):
        print(sequence.device, segment_label.device, self.token.device,
              self.position.device, self.segment.device)
        x = self.token(sequence) + self.position(sequence) + \
            self.segment(segment_label)
        return self.dropout(x)
