import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights
from rvc.lib.algorithm.residuals import ResBlock, apply_mask


class SineGeneratorSeparate(nn.Module):
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
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = (
                self._generate_sine_wave(f0, upsampling_factor) * self.sine_amplitude
            )
            noise = torch.randn_like(sine_waves) * self.noise_stddev
            return sine_waves, noise


class SourceModuleHnSeparate(nn.Module):
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

    def forward(self, x: torch.Tensor, upsample_factor: int = 1):
        sine_wavs, noise_wavs = self.l_sin_gen(x, upsample_factor)

        dtype = self.l_linear_s.weight.dtype
        sine_wavs = sine_wavs.to(dtype=dtype)
        noise_wavs = noise_wavs.to(dtype=dtype)

        sine_merge = self.l_tanh(self.l_linear_s(sine_wavs))
        noise_merge = self.l_tanh(self.l_linear_n(noise_wavs))

        return sine_merge.transpose(1, 2), noise_merge.transpose(1, 2)


class PeriodicityEstimator(nn.Module):
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

    def forward(self, x):
        return self.layers(x)

    def remove_weight_norm(self):
        for m in self.layers:
            if isinstance(m, nn.Conv1d):
                remove_weight_norm(m)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        return F.layer_norm(
            x.transpose(1, 2), self.normalized_shape, self.weight, self.bias, self.eps
        ).transpose(1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, expansion=4, drop_path=0.0):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = LayerNorm(dim)
        self.pw_conv1 = nn.Conv1d(dim, dim * expansion, kernel_size=1)
        self.act = nn.SiLU()
        self.pw_conv2 = nn.Conv1d(dim * expansion, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = input + self.drop_path(x)
        return x


class EvaResBlock(ResBlock):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilations: Tuple[int] = (1, 3, 5)
    ):
        super().__init__(channels, kernel_size, dilations)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt_residual = x
            x = F.silu(x)
            x = apply_mask(x, x_mask)
            x = c1(x)
            x = F.silu(x)
            x = apply_mask(x, x_mask)
            x = c2(x)
            x = x + xt_residual
        return apply_mask(x, x_mask)


class ContextAwareModule(nn.Module):
    def __init__(
        self, dims=[128, 256, 384, 512], depths=[3, 3, 9, 3], drop_path_rate=0.2
    ):
        super().__init__()

        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        current_dim = dims[0]
        idx = 0

        for i, depth in enumerate(depths):
            blocks = []
            for j in range(depth):
                blocks.append(ConvNeXtBlock(dim=current_dim, drop_path=dpr[idx + j]))
            self.stages.append(nn.Sequential(*blocks))

            if i < len(dims) - 1 and dims[i + 1] != current_dim:
                self.stages.append(
                    nn.Sequential(
                        LayerNorm(current_dim),
                        nn.Conv1d(current_dim, dims[i + 1], kernel_size=1),
                    )
                )
                current_dim = dims[i + 1]

            idx += depth
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class EvaGanNSFGenerator(nn.Module):
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
        cam_depths: List[int] = [3, 3, 9, 3],
        cam_dims: List[int] = [128, 256, 384, 512],
        drop_path_rate: float = 0.2,
    ):
        super(EvaGanNSFGenerator, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing

        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnSeparate(sample_rate=sr, harmonic_num=0)

        self.conv_pre = nn.Conv1d(initial_channel, cam_dims[0], 7, 1, padding=3)
        self.norm_pre = LayerNorm(cam_dims[0])

        self.periodicity_estimator = PeriodicityEstimator(
            in_channels=cam_dims[0], residual_channels=64, conv_layers=3
        )

        self.cam = ContextAwareModule(cam_dims, cam_depths, drop_path_rate)

        assert cam_dims[-1] == upsample_initial_channel, (
            f"CAM out dim {cam_dims[-1]} must equal upsample_initial_channel {upsample_initial_channel}"
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
                EvaResBlock(channels[i], k, d)
                for i in range(len(self.ups))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

        self.conv_post = nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        har_source, noise_source = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        noise_source = noise_source.transpose(1, 2)

        x = self.conv_pre(x)
        x = self.norm_pre(x)

        mask_periodicity = self.periodicity_estimator(x)

        if self.training and self.checkpointing:
            x = checkpoint(self.cam, x, use_reentrant=False)
        else:
            x = self.cam(x)

        if g is not None:
            x = x + self.cond(g)

        for i, (ups, sine_conv, noise_conv) in enumerate(
            zip(self.ups, self.sine_convs, self.noise_convs)
        ):
            x = F.silu(x)
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

        x = F.silu(x)
        x = torch.tanh(self.conv_post(x))

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        self.periodicity_estimator.remove_weight_norm()

    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        self.periodicity_estimator.remove_weight_norm()
        return self
