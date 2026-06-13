import types
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class _Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class _PatchTSTModel(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        padding = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

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
            norm_layer=nn.Sequential(
                _Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                _Transpose(1, 2),
            ),
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Patching: (B, T, C) → (B, C, T) for patch_embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, _ = self.encoder(enc_out)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)   # (B, nvars, d_model, patch_num)

        # Classification head
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output


class PatchTST(nn.Module):
    """PatchTST wrapper for MVMS-net (multi-label ECG classification).

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
            d_model=128,
            d_ff=256,
            n_heads=16,
            e_layers=3,
            dropout=dropout,
            factor=1,
            activation='gelu',
        )
        self.model = _PatchTSTModel(c)

    def forward(self, x):
        # x: (B, 12, T) → permute to (B, T, 12)
        return self.model(x.permute(0, 2, 1), None, None, None)
