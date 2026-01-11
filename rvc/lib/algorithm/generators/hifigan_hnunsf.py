import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights
from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock


class SineGeneratorSeparate(nn.Module):
    """
    Waveform generator that synthesizes harmonic (sine) and aperiodic (noise) components separately.

    This module provides the raw source signals required by the HN-uSFGAN architecture.
    Unlike standard NSF, it does not mix the signals internally, allowing the downstream
    neural network to perform dynamic mixing based on periodicity estimation.

    Args:
        sampling_rate (int): Audio sampling rate in Hz.
        num_harmonics (int): Number of harmonic overtones to generate above the fundamental.
        sine_amplitude (float): Base amplitude scaling for the sine waves.
        noise_stddev (float): Standard deviation for the Gaussian noise generator.
    """

    def __init__(
        self,
        sampling_rate: int,
        num_harmonics: int = 0,
        sine_amplitude: float = 0.1,
        noise_stddev: float = 0.003,
    ):
        super(SineGeneratorSeparate, self).__init__()
        self.sampling_rate = sampling_rate
        self.num_harmonics = num_harmonics
        self.sine_amplitude = sine_amplitude
        self.noise_stddev = noise_stddev
        self.waveform_dim = self.num_harmonics + 1

    def _generate_sine_wave(
        self, f0: torch.Tensor, upsampling_factor: int
    ) -> torch.Tensor:
        """
        Generates harmonic sine waves from a fundamental frequency sequence.

        Args:
            f0 (Tensor): Fundamental frequency tensor of shape (B, T, 1).
            upsampling_factor (int): Factor by which the time dimension matches the audio resolution.

        Returns:
            Tensor: Generated sine waves of shape (B, T * upsample, harmonics + 1).
        """
        batch_size, length, _ = f0.shape
        upsampling_grid = torch.arange(
            1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device
        )

        phase_increments = (f0 / self.sampling_rate) * upsampling_grid
        phase_remainder = torch.fmod(phase_increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0).to(f0.dtype)
        phase_increments += F.pad(cumulative_phase, (0, 0, 1, 0), mode="constant")

        phase_increments = phase_increments.reshape(batch_size, -1, 1)

        harmonic_scale = torch.arange(
            1, self.waveform_dim + 1, dtype=f0.dtype, device=f0.device
        ).reshape(1, 1, -1)
        phase_increments *= harmonic_scale

        random_phase = torch.rand(1, 1, self.waveform_dim, device=f0.device)
        random_phase[..., 0] = 0
        phase_increments += random_phase

        sine_waves = torch.sin(2 * math.pi * phase_increments)
        return sine_waves

    def forward(
        self, f0: torch.Tensor, upsampling_factor: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate source signals.

        Args:
            f0 (Tensor): Input fundamental frequency (B, T, 1).
            upsampling_factor (int): Upsampling scale factor.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - sine_waves: The harmonic component tensor.
                - noise: The aperiodic component tensor.
        """
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = (
                self._generate_sine_wave(f0, upsampling_factor) * self.sine_amplitude
            )
            noise = torch.randn_like(sine_waves) * self.noise_stddev
            return sine_waves, noise


class SourceModuleHnSeparate(nn.Module):
    """
    Harmonic-plus-Noise (HN) Source Module.

    This module wraps the waveform generator and projects the generated sine and noise
    signals into the hidden feature space using separate linear layers. It handles
    mixed-precision type casting to ensure stability during training.

    Args:
        sample_rate (int): Audio sampling rate.
        harmonic_num (int): Number of harmonics.
        sine_amp (float): Sine amplitude.
        add_noise_std (float): Noise standard deviation.
    """

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
    ):
        super(SourceModuleHnSeparate, self).__init__()
        self.l_sin_gen = SineGeneratorSeparate(
            sample_rate, harmonic_num, sine_amp, add_noise_std
        )
        self.l_linear_s = nn.Linear(harmonic_num + 1, 1)
        self.l_linear_n = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(
        self, x: torch.Tensor, upsample_factor: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates and projects excitation features.

        Args:
            x (Tensor): F0 sequence.
            upsample_factor (int): Upsampling factor.

        Returns:
            Tuple[Tensor, Tensor]:
                - Harmonic features of shape (B, C, T).
                - Noise features of shape (B, C, T).
        """
        sine_wavs, noise_wavs = self.l_sin_gen(x, upsample_factor)

        dtype = self.l_linear_s.weight.dtype
        sine_wavs = sine_wavs.to(dtype=dtype)
        noise_wavs = noise_wavs.to(dtype=dtype)

        sine_merge = self.l_tanh(self.l_linear_s(sine_wavs))
        noise_merge = self.l_tanh(self.l_linear_n(noise_wavs))

        return sine_merge.transpose(1, 2), noise_merge.transpose(1, 2)


class PeriodicityEstimator(nn.Module):
    """
    Periodicity Estimator Network.

    Based on the architecture defined in the HN-uSFGAN paper. This network estimates
    a soft mask (V/UV decision) from the input features, determining the ratio
    between harmonic and noise components in the final generation.

    Architecture consists of a stack of 1D convolutions with ReLU activations,
    ending with a Sigmoid activation to bound the output between 0 and 1.

    Args:
        in_channels (int): Input feature channels.
        residual_channels (int): Hidden channels for convolution layers.
        conv_layers (int): Number of convolutional layers.
        kernel_size (int): Convolution kernel size.
        dilation (int): Dilation factor.
        padding_mode (str): Padding mode for convolutions.
    """

    def __init__(
        self,
        in_channels,
        residual_channels=64,
        conv_layers=3,
        kernel_size=5,
        dilation=1,
        padding_mode="replicate",
    ):
        super(PeriodicityEstimator, self).__init__()

        modules = []
        for idx in range(conv_layers):
            out_ch = residual_channels if idx != conv_layers - 1 else 1

            conv1d = weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=kernel_size // 2 * dilation,
                    padding_mode=padding_mode,
                )
            )

            if idx != conv_layers - 1:
                nonlinear = nn.ReLU(inplace=True)
            else:
                nn.init.normal_(conv1d.weight, std=1e-4)
                nonlinear = nn.Sigmoid()

            modules += [conv1d, nonlinear]
            in_channels = out_ch

        self.layers = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimates periodicity mask.

        Args:
            x (Tensor): Input features (B, C, T).

        Returns:
            Tensor: Periodicity mask 'a' (B, 1, T).
        """
        return self.layers(x)

    def remove_weight_norm(self):
        """Removes weight normalization for inference."""
        for m in self.layers:
            if isinstance(m, nn.Conv1d):
                remove_weight_norm(m)


class HiFiGANNSFGenerator(nn.Module):
    """
    HiFiGAN Generator integrated with the HN-uSFGAN architecture.

    This class implements the Unified Source-Filter GAN with Harmonic-plus-Noise
    source excitation. It utilizes parallel branches for harmonic and noise
    components, which are dynamically mixed at each upsampling stage based on
    a periodicity mask estimated from the input mel-spectrogram.

    Args:
        initial_channel (int): Input channels for the first convolution.
        resblock_kernel_sizes (list): Kernel sizes for residual blocks.
        resblock_dilation_sizes (list): Dilation sizes for residual blocks.
        upsample_rates (list): Upsampling factors.
        upsample_initial_channel (int): Channels after initial convolution.
        upsample_kernel_sizes (list): Kernel sizes for upsampling layers.
        gin_channels (int): Global conditioning channels.
        sr (int): Sampling rate.
        checkpointing (bool): Enable gradient checkpointing.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        checkpointing: bool = False,
    ):
        super(HiFiGANNSFGenerator, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing
        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

        self.m_source = SourceModuleHnSeparate(sample_rate=sr, harmonic_num=0)

        self.conv_pre = weight_norm(
            nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )

        self.periodicity_estimator = PeriodicityEstimator(
            in_channels=upsample_initial_channel, residual_channels=64, conv_layers=3
        )

        self.ups = nn.ModuleList()
        self.sine_convs = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            if u % 2 == 0:
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2

            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        channels[i],
                        k,
                        u,
                        padding=padding,
                        output_padding=u % 2,
                    )
                )
            )

            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding_src = 0 if stride == 1 else (kernel - stride) // 2

            self.sine_convs.append(
                nn.Conv1d(
                    1,
                    channels[i],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding_src,
                )
            )
            self.noise_convs.append(
                nn.Conv1d(
                    1,
                    channels[i],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding_src,
                )
            )

        self.resblocks = nn.ModuleList(
            [
                ResBlock(channels[i], k, d)
                for i in range(len(self.ups))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

        self.conv_post = weight_norm(
            nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        )
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for audio generation.

        Args:
            x (Tensor): Input mel-spectrogram of shape (B, Mel_Dim, T_mel).
            f0 (Tensor): Fundamental frequency of shape (B, T_mel, 1).
            g (Tensor, optional): Global embedding of shape (B, Gin_Dim, 1).

        Returns:
            Tensor: Generated audio waveform of shape (B, 1, T_audio).
        """
        har_source, noise_source = self.m_source(f0, self.upp)

        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        mask_periodicity = self.periodicity_estimator(x)

        for i, (ups, sine_conv, noise_conv) in enumerate(
            zip(self.ups, self.sine_convs, self.noise_convs)
        ):
            x = F.leaky_relu(x, self.lrelu_slope)

            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
            else:
                x = ups(x)

            s_feat = sine_conv(har_source)
            n_feat = noise_conv(noise_source)

            m_curr = F.interpolate(
                mask_periodicity, size=x.shape[-1], mode="linear", align_corners=True
            )

            excitation = (m_curr * s_feat) + ((1 - m_curr) * n_feat)
            x = x + excitation

            if self.training and self.checkpointing:
                xs = sum(
                    [
                        checkpoint(resblock, x, use_reentrant=False)
                        for j, resblock in enumerate(self.resblocks)
                        if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)
                    ]
                )
            else:
                xs = sum(
                    [
                        resblock(x)
                        for j, resblock in enumerate(self.resblocks)
                        if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)
                    ]
                )
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = torch.tanh(self.conv_post(x))

        return x

    def remove_weight_norm(self):
        """Removes weight normalization from all modules for inference."""
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        self.periodicity_estimator.remove_weight_norm()

    def __prepare_scriptable__(self):
        self.remove_weight_norm()
        return self
