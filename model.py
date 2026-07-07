import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, block_size):

        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.fc_out = nn.Linear(embedding_dim, embedding_dim)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

      batch_size = x.shape[0]
      sequence_length = x.shape[1]

      Q = self.query(x)
      K = self.key(x)
      V = self.value(x)

      Q = Q.view(
    batch_size,
    sequence_length,
    self.num_heads,
    self.head_dim
)
      K = K.view(
    batch_size,
    sequence_length,
    self.num_heads,
    self.head_dim
)

      V = V.view(
    batch_size,
    sequence_length,
    self.num_heads,
    self.head_dim
)

      Q = Q.transpose(1,2)
      K = K.transpose(1,2)
      V = V.transpose(1,2)

      scores = Q @ K.transpose(-2, -1)
      scores = scores / math.sqrt(self.head_dim)
      mask = self.mask[:sequence_length, :sequence_length]
      scores = scores.masked_fill( mask == 0, float("-inf"))
      attention = F.softmax(scores,dim=-1)
      output = attention @ V
      output = output.transpose(1,2)
      output = output.contiguous().view(batch_size,sequence_length,self.embedding_dim)
      output = self.fc_out(output)
      return output

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)
    

class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            block_size
        )

        self.ffn = FeedForward(
            embedding_dim
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    

class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        block_size,
        num_heads,
        num_layers
    ):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        self.blocks = nn.Sequential(

            *[
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    block_size
                )
                for _ in range(num_layers)
            ]

        )

        self.ln = nn.LayerNorm(
            embedding_dim
        )

        self.fc = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

      batch_size, sequence_length = x.shape

      positions = torch.arange(
        sequence_length,
        device=x.device
    )

      token_embeddings = self.token_embedding(x)

      position_embeddings = self.position_embedding(
        positions
    )

      x = token_embeddings + position_embeddings
      x = self.blocks(x)
      x = self.ln(x)
      logits = self.fc(x)
      return logits