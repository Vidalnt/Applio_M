import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE


class HarmonicFilter(nn.Module):
    """
    Vectorized Implementation of the Harmonic Filter (Section III-A).
    Maps Linear STFT -> Log Harmonic Tensor.
    """

    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        f_min=32.7,
        bins_per_octave=24,
        num_harmonics=10,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.B = bins_per_octave

        # List of harmonics: h=0.5 (sub-harmonic) and h=1..H
        # Paper: "one half harmonic (h = 0.5) ... is added"
        self.harmonics = [0.5] + [float(h) for h in range(1, num_harmonics + 1)]
        self.num_total_harmonics = len(self.harmonics)

        # 1. Create linear STFT frequency axis (f)
        # Shape: (1, 1, n_fft/2 + 1) for broadcasting
        self.register_buffer(
            "f_stft", torch.linspace(0, sample_rate / 2, n_fft // 2 + 1).view(1, 1, -1)
        )

        # 1. Create linear STFT frequency axis (f) accurately matching FFT bins
        # Shape: (1, 1, n_fft/2 + 1) for broadcasting
        # Usamos rfftfreq para mayor precisión matemática con la FFT real
        # freqs = torch.fft.rfftfreq(n_fft, d=1/sample_rate)
        # self.register_buffer("f_stft", freqs.view(1, 1, -1).float())

        # 2. Calculate base center frequencies (f_c) using Equation 5
        # Criterion: Highest harmonic must not exceed Nyquist (fs/2)
        # h_max * f_base_max <= fs/2  =>  f_base_max <= fs / (2 * h_max)
        max_h = self.harmonics[-1]
        f_max_base = (sample_rate / 2) / max_h

        # Calculate required number of bins: k = B * log2(f_max / f_min)
        if f_max_base > self.f_min:
            num_bins = int(self.B * np.log2(f_max_base / self.f_min))
        else:
            num_bins = 1  # Fallback to avoid errors with very low SR

        k = torch.arange(num_bins).float()

        # Logarithmic base frequencies. Shape: (1, F_bins, 1)
        self.register_buffer(
            "f_c_base", (self.f_min * (2 ** (k / self.B))).view(1, -1, 1)
        )

        # 3. Learnable Gamma parameter (initialized to 1.0)
        self.gamma_param = nn.Parameter(torch.tensor(1.0))

    def get_filter_bank(self):
        """Generates the dynamic filter bank at each training step."""
        # Constraint: gamma >= 1 (Section III-A)
        gamma = torch.clamp(self.gamma_param, min=1.0)

        filters_list = []
        for h in self.harmonics:
            # Filter center for harmonic h: h * f_c
            f_c_h = h * self.f_c_base

            # Dynamic bandwidth (Equation 9)
            f_bw_h = (0.1079 * f_c_h + 24.7) / gamma

            # Triangular filter (Equation 6)
            # ∇h = [1 - 2|f - h·fc| / bw]_+
            dist = torch.abs(self.f_stft - f_c_h)
            triangle = 1.0 - (2.0 * dist / f_bw_h)
            band_filter = torch.relu(triangle)  # ReLU equivalent to []_+

            filters_list.append(band_filter)

        # Stack all filters: (Num_Harmonics, Out_Log_Bins, In_Linear_Bins)
        return torch.cat(filters_list, dim=0)

    def forward(self, x_stft_mag):
        # x_stft_mag: (Batch, 1, In_Linear_Bins, Time)
        spec = x_stft_mag.squeeze(1)  # (B, In, T)

        # Get current filters (depend on gamma)
        filters = self.get_filter_bank()  # (H, Out, In)

        # Filter application: Interpolation from Linear to Log-Harmonic
        # Using einsum for maximum efficiency and mathematical correctness
        # b:batch, h:harmonic, o:out_bins, i:in_bins, t:time
        harmonic_tensor = torch.einsum("hoi,bit->bhot", filters, spec)

        return harmonic_tensor


class HCB(nn.Module):
    """Hybrid Convolution Block (Fig. 2a): Sum of Depthwise and Normal Conv."""

    def __init__(
        self, in_channels, out_channels, kernel_size=(7, 7), use_spectral_norm=False
    ):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # Depthwise Separable branch
        self.ds_conv = nn.Sequential(
            norm_f(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    padding=padding,
                    groups=in_channels,
                )
            ),
            nn.LeakyReLU(LRELU_SLOPE),
            norm_f(nn.Conv2d(in_channels, out_channels, 1)),
        )

        # Normal Convolution branch
        self.normal_conv = nn.Sequential(
            norm_f(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)),
            nn.LeakyReLU(
                LRELU_SLOPE
            ),  # Activation not explicit in diagram but standard in implementations
        )

    def forward(self, x):
        return self.ds_conv(x) + self.normal_conv(x)


class MDCBlock(nn.Module):
    """Multi-scale Dilated Convolution (Fig. 2b)."""

    def __init__(self, channels=32, kernel_size=(5, 5), use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        # 3 parallel branches with dilations 1, 2, 4
        self.dilated_convs = nn.ModuleList()
        for d in [1, 2, 4]:
            pad_h = d * (kernel_size[0] - 1) // 2
            pad_w = d * (kernel_size[1] - 1) // 2
            self.dilated_convs.append(
                nn.Sequential(
                    norm_f(
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=d,
                            padding=(pad_h, pad_w),
                        )
                    ),
                    nn.LeakyReLU(LRELU_SLOPE),
                )
            )

        # Final conv with stride (2, 1) to reduce frequency dimension
        self.final_conv = nn.Sequential(
            norm_f(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size,
                    stride=(2, 1),
                    padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                )
            ),
            nn.LeakyReLU(LRELU_SLOPE),
        )

    def forward(self, x):
        # Sum of parallel branches
        out_sum = 0
        for layer in self.dilated_convs:
            out_sum += layer(x)

        # Downsample
        return self.final_conv(out_sum)


class UnivHDDiscriminator(nn.Module):
    """
    Complete Universal Harmonic Discriminator for integration into RVC.
    """

    def __init__(
        self,
        sample_rate,
        n_fft,
        hop_length,
        win_length,
        num_harmonics=10,
        use_spectral_norm=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate

        self.register_buffer("window", torch.hann_window(win_length))

        # 1. Harmonic Filter
        self.harmonic_filter = HarmonicFilter(
            sample_rate=sample_rate, n_fft=n_fft, num_harmonics=num_harmonics
        )

        # Channels = H + 1 (for the half-harmonic)
        in_channels = self.harmonic_filter.num_total_harmonics
        base_channels = 32

        # 2. HCB
        self.hcb = HCB(
            in_channels,
            base_channels,
            kernel_size=(7, 7),
            use_spectral_norm=use_spectral_norm,
        )

        # 3. MDCs (Features are collected after each block)
        self.mdc1 = MDCBlock(
            base_channels, kernel_size=(5, 5), use_spectral_norm=use_spectral_norm
        )
        self.mdc2 = MDCBlock(
            base_channels, kernel_size=(5, 5), use_spectral_norm=use_spectral_norm
        )
        self.mdc3 = MDCBlock(
            base_channels, kernel_size=(5, 5), use_spectral_norm=use_spectral_norm
        )

        # 4. Final Layer
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.final_conv = nn.Sequential(
            norm_f(nn.Conv2d(base_channels, 1, kernel_size=(3, 3), padding=(1, 1))),
            nn.Flatten(),
        )

    def forward(self, x):
        # x: Waveform (Batch, 1, Time)

        # Robust internal STFT
        pad_size = int((self.n_fft - self.hop_length) / 2)
        x_pad = F.pad(x.squeeze(1), (pad_size, pad_size), mode="reflect")

        x_stft = torch.stft(
            x_pad,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window,
            center=False,  # Important for strict alignment
            return_complex=True,
        )
        mag = torch.abs(x_stft).unsqueeze(1)  # (B, 1, F, T)

        # UnivHD flow
        h_tensor = self.harmonic_filter(mag)

        fmap = []

        feat = self.hcb(h_tensor)
        fmap.append(feat)

        feat = self.mdc1(feat)
        fmap.append(feat)

        feat = self.mdc2(feat)
        fmap.append(feat)

        feat = self.mdc3(feat)
        fmap.append(feat)

        # Final score (Adversarial Loss)
        score = self.final_conv(feat)
        fmap.append(score)

        return score, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-period discriminator.

    This class implements a multi-period discriminator, which is used to
    discriminate between real and fake audio signals. The discriminator
    is composed of a series of convolutional layers that are applied to
    the input signal at different periods.

    Args:
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        checkpointing: bool = False,
        version: str = "v2",
        *kwargs,
    ):
        super().__init__()

        if version == "v1":
            periods = [2, 3, 5, 7, 11, 17]
            resolutions = []
        elif version == "v2":
            periods = [2, 3, 5, 7, 11, 17, 23, 37]
            resolutions = []
        elif version == "v3":
            periods = [2, 3, 5, 7, 11]
            resolutions = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]

        self.checkpointing = checkpointing
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
            + [
                DiscriminatorR(r, use_spectral_norm=use_spectral_norm)
                for r in resolutions
            ]
        )
        if version == "v3":
            self.discriminators.append(
                UnivHDDiscriminator(use_spectral_norm=use_spectral_norm, *kwargs)
            )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            if self.training and self.checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(d, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """
    Discriminator for the short-term component.

    This class implements a discriminator for the short-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorP(torch.nn.Module):
    """
    Discriminator for the long-term component.

    This class implements a discriminator for the long-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal at a given
    period.

    Args:
        period (int): Period of the discriminator.
        kernel_size (int): Kernel size of the convolutional layers. Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorR(torch.nn.Module):
    def __init__(self, resolution, use_spectral_norm=False):
        super().__init__()

        self.resolution = resolution
        self.lrelu_slope = 0.1
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        1,
                        32,
                        (3, 9),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x).unsqueeze(1)

        for layer in self.convs:
            x = F.leaky_relu(layer(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(
            x,
            (pad, pad),
            mode="reflect",
        ).squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.ones(win_length, device=x.device),
            center=False,
            return_complex=True,
        )

        mag = torch.norm(torch.view_as_real(x), p=2, dim=-1)  # [B, F, TT]

        return mag
