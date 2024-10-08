import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import aerosandbox as asb

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, channels)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1)  # (batch_size, channels, sequence_length)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            #nn.GroupNorm(1, mid_channels),
            nn.InstanceNorm1d(mid_channels), # yayati suggestion
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            #nn.GroupNorm(1, out_channels),
            nn.InstanceNorm1d(out_channels), # yayati suggestion
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        if x.shape[2] != skip_x.shape[2]:
            x = F.interpolate(x, size=skip_x.shape[2], mode="nearest")
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb

class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, base_dim=8, dim_mults=[1,2,4,8], device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, base_dim*dim_mults[0])
        self.down1 = Down(base_dim*dim_mults[0], base_dim*dim_mults[1], time_dim)
        self.sa1 = SelfAttention(base_dim*dim_mults[1])
        self.down2 = Down(base_dim*dim_mults[1], base_dim*dim_mults[2], time_dim)
        self.sa2 = SelfAttention(base_dim*dim_mults[2])
        self.down3 = Down(base_dim*dim_mults[2], base_dim*dim_mults[2], time_dim)
        self.sa3 = SelfAttention(base_dim*dim_mults[2])

        self.bot1 = DoubleConv(base_dim*dim_mults[2], base_dim*dim_mults[3])
        self.bot2 = DoubleConv(base_dim*dim_mults[3], base_dim*dim_mults[3])
        self.bot3 = DoubleConv(base_dim*dim_mults[3], base_dim*dim_mults[2])

        self.up1 = Up(base_dim*dim_mults[3], base_dim*dim_mults[1], time_dim)
        self.sa4 = SelfAttention(base_dim*dim_mults[1])
        self.up2 = Up(base_dim*dim_mults[2], base_dim*dim_mults[0], time_dim)
        self.sa5 = SelfAttention(base_dim*dim_mults[0])
        self.up3 = Up(base_dim*dim_mults[1], base_dim*dim_mults[0], time_dim)
        self.sa6 = SelfAttention(base_dim*dim_mults[0])
        self.outc = nn.Conv1d(base_dim*dim_mults[0], c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output




class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=128, cond_dim=10, base_dim=8, dim_mults=[1,2,4,8], device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, base_dim*dim_mults[0])
        self.down1 = Down(base_dim*dim_mults[0], base_dim*dim_mults[1], time_dim)
        self.sa1 = SelfAttention(base_dim*dim_mults[1])
        self.down2 = Down(base_dim*dim_mults[1], base_dim*dim_mults[2], time_dim)
        self.sa2 = SelfAttention(base_dim*dim_mults[2])
        self.down3 = Down(base_dim*dim_mults[2], base_dim*dim_mults[2], time_dim)
        self.sa3 = SelfAttention(base_dim*dim_mults[2])

        self.bot1 = DoubleConv(base_dim*dim_mults[2], base_dim*dim_mults[3])
        self.bot2 = DoubleConv(base_dim*dim_mults[3], base_dim*dim_mults[3])
        self.bot3 = DoubleConv(base_dim*dim_mults[3], base_dim*dim_mults[2])

        self.up1 = Up(base_dim*dim_mults[3], base_dim*dim_mults[1], time_dim)
        self.sa4 = SelfAttention(base_dim*dim_mults[1])
        self.up2 = Up(base_dim*dim_mults[2], base_dim*dim_mults[0], time_dim)
        self.sa5 = SelfAttention(base_dim*dim_mults[0])
        self.up3 = Up(base_dim*dim_mults[1], base_dim*dim_mults[0], time_dim)
        self.sa6 = SelfAttention(base_dim*dim_mults[0])
        self.outc = nn.Conv1d(base_dim*dim_mults[0], c_out, kernel_size=1)

        self.cond_1 = nn.Linear(cond_dim, 8) # yayati suggestion
        self.cond_2 = nn.Linear(8, 64) # yayati suggestion
        self.cond_emb = nn.Bilinear(64, time_dim, time_dim) # yayati suggestion
        #self.cond_emb = nn.Bilinear(cond_dim, time_dim, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            y = self.cond_1(y)
            y = F.relu(y)
            y = self.cond_2(y)
            y = self.cond_emb(y, t)
            t = t + y  # No need to unsqueeze here since pos_encoding and label_emb already have same dimensions

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    








if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet_conditional(c_in=1, c_out=1, cond_dim=1, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 1, 200) 
    t = torch.tensor([500] * x.shape[0]).long()
    y = torch.tensor([1] * x.shape[0]).float().unsqueeze(-1)
    print(net(x, t, y).shape)
