import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class _iTransformerModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model,
            configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output


class iTransformer(nn.Module):
    """iTransformer wrapper for MVMS-net (multi-label ECG classification).

    Input:  (B, 12, T)   – MVMS-net convention
    Output: (B, num_classes)
    """

    def __init__(self, num_classes):
        super().__init__()
        try:
            from config import config as _cfg
            seq_len = getattr(_cfg, 'seq_len', 1000)
            dropout = getattr(_cfg, 'dropout', 0.1)
        except Exception:
            seq_len, dropout = 1000, 0.1

        c = types.SimpleNamespace(
            task_name='classification',
            seq_len=seq_len,
            pred_len=0,
            enc_in=12,
            num_class=num_classes,
            d_model=256,
            d_ff=256,
            n_heads=8,
            e_layers=4,
            dropout=dropout,
            factor=1,
            activation='gelu',
            embed='timeF',
            freq='h',
        )
        self.model = _iTransformerModel(c)

    def forward(self, x):
        # x: (B, 12, T) → permute to (B, T, 12)
        return self.model(x.permute(0, 2, 1), None, None, None)
