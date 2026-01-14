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
            smoothing_lifter[1:] = torch.sin(arg[1:]) / arg[1:]
            compensation_lifter[0] = self.q0 + 2.0 * self.q1
            compensation_lifter[1:] = self.q0 + 2.0 * self.q1 * torch.cos(2.0 * arg[1:])
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
        sample_rate=40000,
        fft_size=2048,
        hop_size=400,
        f0_floor=40,
        f0_ceil=1100,
        n_mels=125,
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
        if s.dim() == 3:
            s = s.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        if f.dim() == 3:
            f = f.squeeze(1)
        elif f.dim() == 2 and f.size(1) == 1:
            f = f.squeeze(1)

        with torch.no_grad():
            envelope_log = self.cheaptrick.forward(y, f, self.power, self.elim_0th)
            y_mag = stft(
                y,
                self.fft_size,
                self.hop_size,
                self.win_length,
                self.window,
                power=self.power,
            )

            minlen = min(envelope_log.size(1), y_mag.size(1))
            envelope_log = envelope_log[:, :minlen, :]
            y_mag = y_mag[:, :minlen, :]

            y_log = torch.log(torch.clamp(y_mag, min=1e-7))
            target_residual = torch.exp(y_log - envelope_log)

            if self.elim_0th:
                y_mean = y_mag.mean(dim=-1, keepdim=True)
                t_mean = target_residual.mean(dim=-1, keepdim=True)
                target_residual = target_residual * (y_mean / (t_mean + 1e-7))

            target_mel = torch.matmul(target_residual, self.melmat)
            target_log_mel = torch.log(torch.clamp(target_mel, min=1e-7))

        s_mag = stft(
            s,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            power=self.power,
        )

        minlen = min(minlen, s_mag.size(1))
        s_mag = s_mag[:, :minlen, :]
        target_log_mel = target_log_mel[:, :minlen, :]

        s_mel = torch.matmul(s_mag, self.melmat)
        s_log_mel = torch.log(torch.clamp(s_mel, min=1e-7))

        loss = F.l1_loss(s_log_mel, target_log_mel.detach())

        return loss
