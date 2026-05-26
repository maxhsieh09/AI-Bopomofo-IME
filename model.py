import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)] # Bug fixed: batch and seq dimension is reversed!
        return self.dropout(x)


class IMEModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        ffn_size=1024,
        max_len=96,
        dropout=0.1,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        #self.pos_embed = nn.Embedding(max_len, hidden_size)
        self.pos_embed = PositionalEncoding(hidden_size, dropout, max_len)

        self.embed_layer_norm = nn.LayerNorm(hidden_size)
        #self.embed_dropout = nn.Dropout(dropout)

        # Don't use nn.TransformerEncoder for initializing different parameters
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ffn_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True
            ) for _ in range(num_layers)]
        )

        self.lm_head = nn.Linear(hidden_size, vocab_size)

        self.apply(self._init_weights)

        self.lm_head.weight = self.token_embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.2)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [B, L]
        attention_mask: [B, L] (1 = valid, 0 = padding)
        """

        #B, L = input_ids.shape
        #device = input_ids.device

        #pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        x = self.pos_embed(self.token_embed(input_ids))
        #x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        x = self.embed_layer_norm(x)
        #x = self.embed_dropout(x)

        # TransformerEncoder uses True = masked, so invert
        key_padding_mask = attention_mask == 0

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)

        logits = self.lm_head(x)  # [B, L, vocab_size]
        return logits
