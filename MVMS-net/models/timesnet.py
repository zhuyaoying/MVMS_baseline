import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def _fft_for_period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class _TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = _fft_for_period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]],
                                      device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


class _TimesNetModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.model = nn.ModuleList([_TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        # x_mark_enc is all-ones so mask is a no-op
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output


class TimesNet(nn.Module):
    """TimesNet wrapper for MVMS-net (multi-label ECG classification).

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
            label_len=0,
            pred_len=0,
            enc_in=12,
            num_class=num_classes,
            d_model=64,
            d_ff=64,
            e_layers=2,
            top_k=3,
            num_kernels=6,
            dropout=dropout,
            embed='timeF',
            freq='h',
            c_out=1,
        )
        self.model = _TimesNetModel(c)

    def forward(self, x):
        # x: (B, 12, T) → permute to (B, T, 12)
        x = x.permute(0, 2, 1)
        B, T, _ = x.shape
        x_mark = torch.ones(B, T, device=x.device)
        return self.model(x, x_mark, None, None)
