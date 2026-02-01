import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from rvc.lib.algorithm.commons import get_padding, init_weights


class TorchSTFT(torch.nn.Module):
    def __init__(
        self, filter_length=800, hop_length=200, win_length=800, window="hann"
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(np.hanning(win_length).astype(np.float32))

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(input_data.device),
            return_complex=True,
        )
        spec = torch.view_as_real(forward_transform)
        return spec[..., 0], spec[..., 1]

    def inverse(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device),
        )
        return inverse_transform.unsqueeze(-2)


class ISTFT(nn.Module):
    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )

        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]
        output_size = (spec.size(-1) - 1) * self.hop_length + self.win_length

        pad = (self.win_length - self.hop_length) // 2
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        window_sq = self.window.square().expand(1, spec.size(-1), -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        y = y / (window_envelope + 1e-11)
        return y


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlockRVC(nn.Module):
    def __init__(
        self, dim, intermediate_dim, layer_scale_init_value=1e-6, gin_channels=0
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.gin_channels = gin_channels
        if gin_channels > 0:
            self.cond_layer = nn.Linear(gin_channels, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, g=None):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        if g is not None and self.gin_channels > 0:
            g_proj = self.cond_layer(g.squeeze(-1))
            x = self.norm(x) + g_proj.unsqueeze(1)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        x = residual + x
        return x


class Mel2FramesRVC(nn.Module):
    def __init__(
        self,
        input_channels,
        dim,
        istft_nfft,
        intermediate_dim,
        num_layers=4,
        gin_channels=0,
    ):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlockRVC(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    gin_channels=gin_channels,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.mag_out = nn.Sequential(
            nn.ReflectionPad1d(1),
            weight_norm(
                nn.Conv1d(
                    dim, istft_nfft // 2 + 1, kernel_size=3, padding=0, bias=False
                )
            ),
        )
        self.phase_out = nn.Sequential(
            nn.ReflectionPad1d(1),
            weight_norm(
                nn.Conv1d(
                    dim, istft_nfft // 2 + 1, kernel_size=3, padding=0, bias=False
                )
            ),
        )

    def forward(self, x, g=None):
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x, g=g)
        x = self.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        mag = self.mag_out(x)
        phase = self.phase_out(x)
        return mag, phase


class SCNetResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(SCNetResBlock, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)
        self.alpha1 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))]
        )
        self.alpha2 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))]
        )
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.alpha1, self.alpha2):
            xt = x + (1 / (a1 + self.no_div_by_zero)) * (torch.sin(a1 * x) ** 2)
            xt = c1(xt)
            xt = xt + (1 / (a2 + self.no_div_by_zero)) * (torch.sin(a2 * xt) ** 2)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs))]
        )
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        for c, a in zip(self.convs, self.alpha):
            xt = x + (1 / (a + self.no_div_by_zero)) * (torch.sin(a * x) ** 2)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class SCNetGenerator(nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        sr=40000,
        gen_istft_n_fft=16,
        gen_istft_hop_size=4,
        har_istft_n_fft=256,
        har_istft_hop_size=64,
        proj_channels=256,
        inter_dim=768,
        scnet_num_layers=4,
    ):
        super(SCNetGenerator, self).__init__()
        self.h = type("h", (object,), {"audio_limit": 0.99})()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # 1. Main Input Conv
        self.conv_pre = weight_norm(
            nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        # 2. Condition Network (Subband)
        # input: initial_channel (192) + f0 (1) = 193
        self.m_source = Mel2FramesRVC(
            input_channels=initial_channel + 1,
            dim=proj_channels,
            istft_nfft=har_istft_n_fft,
            intermediate_dim=inter_dim,
            num_layers=scnet_num_layers,
            gin_channels=gin_channels,
        )

        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        # 3. Build Upsampling Layers
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

            c_cur = upsample_initial_channel // (2 ** (i + 1))

            # Coupling Blocks (Condition injection logic)
            if i + 1 < len(upsample_rates):
                stride_f0 = 2
                self.noise_convs.append(
                    nn.Conv1d(
                        gen_istft_n_fft + 2,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(ResBlock2(c_cur, 7, [1, 3]))
            else:
                self.noise_convs.append(
                    nn.Conv1d(gen_istft_n_fft // 4 + 2, c_cur, kernel_size=1)
                )
                self.noise_res.append(ResBlock2(c_cur, 11, [1, 3]))

            for j, (k_res, d_res) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(SCNetResBlock(c_cur, k_res, d_res))

        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(
            nn.Conv1d(c_cur, self.post_n_fft + 2, 7, 1, padding=3)
        )
        self.reflection_pad = torch.nn.ReflectionPad1d(
            (1, 0)
        )  # Restaurado del original

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        # STFTs
        self.stft_main = TorchSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )
        self.stft_aux = TorchSTFT(
            filter_length=gen_istft_n_fft // 4,
            hop_length=gen_istft_hop_size // 4,
            win_length=gen_istft_n_fft // 4,
        )
        self.m_istft_subband = ISTFT(
            n_fft=har_istft_n_fft,
            hop_length=har_istft_hop_size,
            win_length=har_istft_n_fft,
            padding="same",
        )

    def forward(self, x, f0, g=None):
        if f0 is not None:
            f0_log = torch.log(f0 + 1e-5) / torch.log(torch.tensor(10.0))
            if f0_log.shape[-1] != x.shape[-1]:
                f0_log = F.interpolate(f0_log, size=x.shape[-1], mode="linear")
            x_cond = torch.cat([x, f0_log], dim=1)
        else:
            x_cond = x

        # 1. CondNet Inference
        s_mag, s_phase = self.m_source(x_cond, g)

        s_mag = torch.exp(s_mag).clip(max=1e2)
        source_complex = s_mag * (torch.cos(s_phase) + 1j * torch.sin(s_phase))
        har_source = self.m_istft_subband(source_complex)

        # 2. Prepare Subband Conditions (Multi-res STFT)
        har_spec, har_phase = self.stft_main.transform(har_source.squeeze(1))
        har = torch.cat([har_spec, har_phase], dim=1)

        har_spec1, har_phase1 = self.stft_aux.transform(har_source.squeeze(1))
        har1 = torch.cat([har_spec1, har_phase1], dim=1)

        # 3. Backbone Inference
        x_feat = self.conv_pre(x)
        if g is not None:
            x_feat = x_feat + self.cond(g)

        for i in range(self.num_upsamples):
            x_feat = F.leaky_relu(x_feat, 0.1)

            # Injection logic
            if i == 0:
                x_source = self.noise_convs[i](har)
            else:
                x_source = self.noise_convs[i](har1)
            x_source = self.noise_res[i](x_source)

            x_feat = self.ups[i](x_feat)

            if i == self.num_upsamples - 1:
                x_feat = self.reflection_pad(x_feat)

            x_feat = x_feat + x_source

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x_feat)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x_feat)
            x_feat = xs / self.num_kernels

        x_feat = F.leaky_relu(x_feat)
        x_out = self.conv_post(x_feat)

        spec = torch.exp(x_out[:, : self.post_n_fft // 2 + 1, :])
        phase = x_out[:, self.post_n_fft // 2 + 1 :, :]
        wav_out = self.stft_main.inverse(spec, phase)
        wav_out = torch.clamp(wav_out, -self.h.audio_limit, self.h.audio_limit)

        return wav_out, har_source.unsqueeze(1)

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
