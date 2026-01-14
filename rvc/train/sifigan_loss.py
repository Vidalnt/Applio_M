import math

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel


def stft(x, fft_size, hop_size, win_length, window, power=False):
    x_stft = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        center=True,
        onesided=True,
        return_complex=True,
    )
    real = x_stft.real
    imag = x_stft.imag

    if power:
        return torch.clamp(real**2 + imag**2, min=1e-7).transpose(2, 1)
    else:
        return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class AdaptiveWindowing(nn.Module):
    def __init__(self, sample_rate, hop_size, fft_size, f0_floor, f0_ceil):
        super(AdaptiveWindowing, self).__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.register_buffer("window", torch.zeros((f0_ceil + 1, fft_size)))
        self.zero_padding = nn.ConstantPad2d((fft_size // 2, fft_size // 2, 0, 0), 0)

        for f0 in range(f0_floor, f0_ceil + 1):
            half_win_len = round(1.5 * self.sample_rate / f0)
            base_index = torch.arange(
                -half_win_len, half_win_len + 1, dtype=torch.int64
            )
            position = base_index / 1.5 / self.sample_rate
            left = fft_size // 2 - half_win_len
            right = fft_size // 2 + half_win_len + 1
            if left < 0 or right > fft_size:
                continue
            window = torch.zeros(fft_size)
            window[left:right] = 0.5 * torch.cos(math.pi * position * f0) + 0.5
            average = torch.sum(window * window).pow(0.5)
            self.window[f0] = window / average

    def forward(self, x, f, power=False):
        x = self.zero_padding(x).unfold(2, self.fft_size, self.hop_size).squeeze(1)
        windows = self.window[f]
        x = torch.abs(torch.fft.rfft(x[:, :-1, :] * windows, dim=2))
        return x.pow(2) if power else x


class AdaptiveLiftering(nn.Module):
    def __init__(self, sample_rate, fft_size, f0_floor, f0_ceil, q1=-0.15):
        super(AdaptiveLiftering, self).__init__()
        self.sample_rate = sample_rate
        self.bin_size = fft_size // 2 + 1
        self.q1 = q1
        self.q0 = 1.0 - 2.0 * q1
        self.register_buffer(
            "smoothing_lifter", torch.zeros((f0_ceil + 1, self.bin_size))
        )
        self.register_buffer(
            "compensation_lifter", torch.zeros((f0_ceil + 1, self.bin_size))
        )

        for f0 in range(f0_floor, f0_ceil + 1):
            smoothing_lifter = torch.zeros(self.bin_size)
            compensation_lifter = torch.zeros(self.bin_size)
            quefrency = (
                torch.arange(1, self.bin_size, dtype=torch.float32) / sample_rate
            )
            smoothing_lifter[0] = 1.0
            arg = math.pi * f0 * quefrency
            smoothing_lifter[1:] = torch.sin(arg) / arg
            compensation_lifter[0] = self.q0 + 2.0 * self.q1
            compensation_lifter[1:] = self.q0 + 2.0 * self.q1 * torch.cos(2.0 * arg)
            self.smoothing_lifter[f0] = smoothing_lifter
            self.compensation_lifter[f0] = compensation_lifter

    def forward(self, x, f, elim_0th=False):
        smoothing_lifter = self.smoothing_lifter[f]
        compensation_lifter = self.compensation_lifter[f]
        tmp = torch.cat((x, torch.flip(x[:, :, 1:-1], [2])), dim=2)
        cepstrum = torch.fft.rfft(torch.log(torch.clamp(tmp, min=1e-7)), dim=2).real
        if elim_0th:
            cepstrum[:, :, 0] = 0
        liftered_cepstrum = cepstrum * smoothing_lifter * compensation_lifter
        x = torch.fft.irfft(liftered_cepstrum, dim=2)[:, :, : self.bin_size]
        return x


class CheapTrick(nn.Module):
    def __init__(self, sample_rate, hop_size, fft_size, f0_floor=40, f0_ceil=1100):
        super(CheapTrick, self).__init__()
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.ada_wind = AdaptiveWindowing(
            sample_rate, hop_size, fft_size, f0_floor, f0_ceil
        )
        self.ada_lift = AdaptiveLiftering(sample_rate, fft_size, f0_floor, f0_ceil)

    def forward(self, x, f, power=False, elim_0th=False):
        voiced = (f > 0).float()
        f_int = voiced * f + (1.0 - voiced) * self.f0_ceil
        f_int = torch.round(
            torch.clamp(f_int, min=self.f0_floor, max=self.f0_ceil)
        ).long()
        x = self.ada_wind(x, f_int, power)
        x = self.ada_lift(x, f_int, elim_0th)
        return x


class ResidualLoss(nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        fft_size=1024,
        hop_size=240,
        f0_floor=50,
        f0_ceil=1100,
        n_mels=80,
        fmin=0,
        fmax=None,
        power=False,
        elim_0th=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = fft_size
        self.elim_0th = elim_0th
        self.power = power
        self.register_buffer("window", torch.hann_window(self.win_length))
        self.cheaptrick = CheapTrick(sample_rate, hop_size, fft_size)
        if fmax is None:
            fmax = sample_rate / 2
        melmat = librosa_mel(
            sr=sample_rate, n_fft=fft_size, n_mels=n_mels, fmin=0, fmax=None
        ).T
        self.register_buffer("melmat", torch.from_numpy(melmat).float())

    def forward(self, s, y, f):
        # s: (B, 1, T), y: (B, 1, T), f: (B, 1, T')
        s, f = s.squeeze(1), f.squeeze(1)

        with torch.no_grad():
            e = self.cheaptrick.forward(y, f, self.power, self.elim_0th)

            y = stft(
                y.squeeze(1),
                self.fft_size,
                self.hop_size,
                self.win_length,
                self.window,
                power=self.power,
            )

            minlen = min(e.size(1), y.size(1))
            e, y = e[:, :minlen, :], y[:, :minlen, :]

            if self.elim_0th:
                y_mean = y.mean(dim=-1, keepdim=True)

            y = torch.log(torch.clamp(y, min=1e-7))
            t = (y - e).exp()

            if self.elim_0th:
                t_mean = t.mean(dim=-1, keepdim=True)
                t = y_mean / (t_mean + 1e-7) * t

            t = torch.matmul(t, self.melmat)
            t = torch.log(torch.clamp(t, min=1e-7))

        s = stft(
            s,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            power=self.power,
        )
        s = s[:, :minlen, :]
        s = torch.matmul(s, self.melmat)
        s = torch.log(torch.clamp(s, min=1e-7))

        return F.l1_loss(s, t.detach())
