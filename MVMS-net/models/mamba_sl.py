"""MambaSL wrapper for MVMS-net.

⚠️  Requires Linux + CUDA + `pip install mamba_ssm causal_conv1d`.
    Importing this module on Windows will raise ImportError.
"""
import types
import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding

try:
    from layers.MambaBlock import Mamba_TimeVariant
    _mamba_available = True
except ImportError as _e:
    import warnings
    warnings.warn(
        f"MambaSL requires `mamba_ssm` (Linux + CUDA only). MambaSL class will not be usable.\nOriginal error: {_e}"
    )
    _mamba_available = False


class _TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, d_kernel=3):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=d_kernel, padding='same',
            padding_mode='replicate', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class _DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, d_kernel=3, seq_len=1000):
        super().__init__()
        self.value_embedding = _TokenEmbedding(c_in=c_in, d_model=d_model, d_kernel=d_kernel)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max(5000, seq_len))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class _MambaSLModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.embedding = _DataEmbedding(
            configs.enc_in, configs.d_model,
            dropout=configs.dropout,
            d_kernel=configs.num_kernels,
            seq_len=configs.seq_len,
        )
        self.mamba = nn.Sequential(
            Mamba_TimeVariant(
                d_model=configs.d_model,
                d_state=configs.d_ff,
                d_conv=configs.d_conv,
                expand=configs.expand,
                timevariant_dt=bool(configs.tv_dt),
                timevariant_B=bool(configs.tv_B),
                timevariant_C=bool(configs.tv_C),
                use_D=bool(configs.use_D),
                device=configs.device,
            ),
            nn.LayerNorm(configs.d_model),
            nn.SiLU(),
        )
        self.out_layer = nn.Sequential(
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.num_class, bias=False),
        )
        nn.init.xavier_uniform_(self.out_layer[1].weight)

        self.attn_weight = nn.Sequential(
            nn.Linear(configs.d_model, configs.n_heads, bias=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Softmax(dim=1),
        )
        for m in self.attn_weight.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(1.0)

    def forward(self, x_enc, x_mark_enc):
        mamba_in = self.embedding(x_enc)
        mamba_out = self.mamba(mamba_in)

        logit_out = self.out_layer(mamba_out)
        logit_out = logit_out * x_mark_enc.unsqueeze(2)

        w_out = self.attn_weight(mamba_out)
        out = (logit_out * w_out).sum(1)
        return out


class MambaSL(nn.Module):
    """MambaSL wrapper for MVMS-net (multi-label ECG classification).

    Input:  (B, 12, T)   – MVMS-net convention
    Output: (B, num_classes)

    ⚠️  Linux + CUDA only.
    """

    def __init__(self, num_classes):
        super().__init__()
        try:
            from config import config as _cfg
            seq_len = getattr(_cfg, 'seq_len', 1000)
            dropout = getattr(_cfg, 'dropout', 0.1)
        except Exception:
            seq_len, dropout = 1000, 0.1

        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        c = types.SimpleNamespace(
            task_name='classification',
            seq_len=seq_len,
            label_len=0,
            pred_len=0,
            enc_in=12,
            num_class=num_classes,
            d_model=64,
            d_ff=64,
            d_conv=4,
            expand=1,
            n_heads=1,
            num_kernels=3,
            tv_dt=1,
            tv_B=1,
            tv_C=1,
            use_D=0,
            dropout=dropout,
            embed='timeF',
            freq='h',
            device=device,
        )
        self.model = _MambaSLModel(c)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, 12, T) → (B, T, 12)
        B, T, _ = x.shape
        x_mark = torch.ones(B, T, device=x.device)
        return self.model(x, x_mark)
