import numpy as np
import torch


def pd_indexing(x, d, dilation, batch_index, ch_index):
    B, C, T = x.size()
    # batch_index = torch.arange(0, B, dtype=torch.long, device=x.device).reshape(B, 1, 1)
    # ch_index = torch.arange(0, C, dtype=torch.long, device=x.device).reshape(1, C, 1)

    dilations = torch.clamp((d * dilation).long(), min=1)

    idx_base = torch.arange(0, T, dtype=torch.long, device=x.device).reshape(1, 1, T)
    idxP = (idx_base - dilations).abs() % T
    idxP = (batch_index, ch_index, idxP)

    idxF = idx_base + dilations
    overflowed = idxF >= T
    idxF[overflowed] = -(idxF[overflowed] % T)
    idxF = (batch_index, ch_index, idxF)

    return x[idxP], x[idxF]


def index_initial(n_batch, n_ch, device=None):
    batch_index = torch.arange(n_batch, dtype=torch.long, device=device).reshape(
        n_batch, 1, 1
    )
    ch_index = torch.arange(n_ch, dtype=torch.long, device=device).reshape(1, n_ch, 1)
    return batch_index, ch_index


def dilated_factor_torch(f0, fs, dense_factor):
    batch_f0 = f0.clone()
    mask = batch_f0 == 0
    batch_f0[mask] = fs / dense_factor
    dilated_factors = (fs / dense_factor) / batch_f0
    return dilated_factors


class SignalGenerator(torch.nn.Module):
    def __init__(self, sample_rate, hop_size, sine_amp=0.1, noise_amp=0.003):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.sine_amp = sine_amp
        self.noise_amp = noise_amp

    def forward(self, f0, upsample_factor):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)

        B, _, T_frames = f0.size()
        T_target = T_frames * self.hop_size

        noise = torch.randn((B, 1, T_target), device=f0.device)

        vuv_frames = (f0 > 0).float()
        vuv = torch.nn.functional.interpolate(vuv_frames, size=T_target, mode="nearest")

        f0_upsampled = torch.nn.functional.interpolate(
            f0, size=T_target, mode="linear", align_corners=False
        )
        radious = (f0_upsampled.double() / self.sample_rate) % 1

        phase = torch.cumsum(radious, dim=2) * 2 * np.pi
        sine = vuv * torch.sin(phase).to(dtype=f0.dtype) * self.sine_amp

        if self.noise_amp > 0:
            noise_amp_mod = vuv * self.noise_amp + (1.0 - vuv) * (self.noise_amp / 3.0)
            noise = noise * noise_amp_mod
            sine = sine + noise

        return sine


class Snake(torch.nn.Module):
    def __init__(self, channels, init=50):
        super(Snake, self).__init__()
        alpha = init * torch.ones(1, channels, 1)
        self.alpha = torch.nn.Parameter(alpha)

    def forward(self, x):
        return x + torch.sin(self.alpha * x).pow(2) / self.alpha


class Conv1d(torch.nn.Conv1d):
    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv1d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 3, 5),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = torch.nn.ModuleList()
        if use_additional_convs:
            self.convs2 = torch.nn.ModuleList()

        for dilation in dilations:
            if nonlinear_activation == "Snake":
                nonlinear = Snake(channels, **nonlinear_activation_params)
            else:
                nonlinear = getattr(torch.nn, nonlinear_activation)(
                    **nonlinear_activation_params
                )

            self.convs1.append(
                torch.nn.Sequential(
                    nonlinear,
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        bias=bias,
                        padding=(kernel_size - 1) // 2 * dilation,
                    ),
                )
            )

            if use_additional_convs:
                if nonlinear_activation == "Snake":
                    nonlinear = Snake(channels, **nonlinear_activation_params)
                else:
                    nonlinear = getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    )
                self.convs2.append(
                    torch.nn.Sequential(
                        nonlinear,
                        torch.nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                )

    def forward(self, x):
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x


class AdaptiveResidualBlock(torch.nn.Module):
    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 2, 4),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.channels = channels
        self.dilations = dilations
        self.nonlinears = torch.nn.ModuleList()
        self.convsC = torch.nn.ModuleList()
        self.convsP = torch.nn.ModuleList()
        self.convsF = torch.nn.ModuleList()
        if use_additional_convs:
            self.convsA = torch.nn.ModuleList()

        for _ in dilations:
            if nonlinear_activation == "Snake":
                self.nonlinears.append(Snake(channels, **nonlinear_activation_params))
            else:
                self.nonlinears.append(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    )
                )

            self.convsC.append(Conv1d1x1(channels, channels, bias=bias))
            self.convsP.append(Conv1d1x1(channels, channels, bias=bias))
            self.convsF.append(Conv1d1x1(channels, channels, bias=bias))

            if use_additional_convs:
                if nonlinear_activation == "Snake":
                    nonlinear = Snake(channels, **nonlinear_activation_params)
                else:
                    nonlinear = getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    )
                self.convsA.append(
                    torch.nn.Sequential(
                        nonlinear,
                        torch.nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                )

    def forward(self, x, d):
        batch_index, ch_index = index_initial(x.size(0), self.channels, x.device)

        for i, dilation in enumerate(self.dilations):
            xt = self.nonlinears[i](x)
            xP, xF = pd_indexing(xt, d, dilation, batch_index, ch_index)
            xt = self.convsC[i](xt) + self.convsP[i](xP) + self.convsF[i](xF)
            if self.use_additional_convs:
                xt = self.convsA[i](xt)
            x = xt + x
        return x


class SiFiGANGenerator(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=[10, 10, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        source_network_params={
            "resblock_kernel_size": 3,
            "resblock_dilations": [[1], [1, 2], [1, 2, 4], [1, 2, 4, 8]],
            "use_additional_convs": True,
        },
        filter_network_params={
            "resblock_kernel_sizes": [3, 5, 7],
            "resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "use_additional_convs": False,
        },
        sample_rate=24000,
        gin_channels=0,
        share_upsamples=False,
        share_downsamples=False,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        dense_factors=[0.5, 1, 4, 8],
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)

        self.num_upsamples = len(upsample_kernel_sizes)
        self.sample_rate = sample_rate
        self.dense_factors = dense_factors
        self.source_network_params = source_network_params
        self.filter_network_params = filter_network_params
        self.share_upsamples = share_upsamples
        self.share_downsamples = share_downsamples

        self.prod_upsample_scales = np.cumprod(upsample_scales)
        initial_fs = sample_rate / np.prod(upsample_scales)
        self.layer_fs = [initial_fs * s for s in self.prod_upsample_scales]

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, channels, 1)

        self.signal_generator = SignalGenerator(
            sample_rate, hop_size=int(np.prod(upsample_scales))
        )

        self.sn = torch.nn.ModuleDict()
        self.fn = torch.nn.ModuleDict()

        self.input_conv = Conv1d(
            in_channels,
            channels,
            kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )

        self.sn["upsamples"] = torch.nn.ModuleList()
        self.fn["upsamples"] = torch.nn.ModuleList()
        self.sn["blocks"] = torch.nn.ModuleList()
        self.fn["blocks"] = torch.nn.ModuleList()

        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]

            self.sn["upsamples"].append(
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                        output_padding=upsample_scales[i] % 2,
                        bias=bias,
                    ),
                )
            )

            if not share_upsamples:
                self.fn["upsamples"].append(
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.ConvTranspose1d(
                            channels // (2**i),
                            channels // (2 ** (i + 1)),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                            output_padding=upsample_scales[i] % 2,
                            bias=bias,
                        ),
                    )
                )

            self.sn["blocks"].append(
                AdaptiveResidualBlock(
                    kernel_size=source_network_params["resblock_kernel_size"],
                    channels=channels // (2 ** (i + 1)),
                    dilations=source_network_params["resblock_dilations"][i],
                    bias=bias,
                    use_additional_convs=source_network_params["use_additional_convs"],
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )
            )

            for j in range(len(filter_network_params["resblock_kernel_sizes"])):
                self.fn["blocks"].append(
                    ResidualBlock(
                        kernel_size=filter_network_params["resblock_kernel_sizes"][j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=filter_network_params["resblock_dilations"][j],
                        bias=bias,
                        use_additional_convs=filter_network_params[
                            "use_additional_convs"
                        ],
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                )

        self.sn["output_conv"] = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                channels // (2**self.num_upsamples),
                out_channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
        )
        self.fn["output_conv"] = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                channels // (2**self.num_upsamples),
                out_channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )

        self.sn["emb"] = Conv1d(
            1,
            channels // (2**self.num_upsamples),
            kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )

        self.sn["downsamples"] = torch.nn.ModuleList()
        for i in reversed(range(len(upsample_kernel_sizes))):
            self.sn["downsamples"].append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        channels // (2 ** (i + 1)),
                        channels // (2**i),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i]
                        - (upsample_kernel_sizes[i] % 2 == 0),
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            )

        if not share_downsamples:
            self.fn["downsamples"] = torch.nn.ModuleList()
            for i in reversed(range(len(upsample_kernel_sizes))):
                self.fn["downsamples"].append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            channels // (2 ** (i + 1)),
                            channels // (2**i),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            padding=upsample_scales[i]
                            - (upsample_kernel_sizes[i] % 2 == 0),
                            bias=bias,
                        ),
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                    )
                )

        self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x, f0, g=None):
        c = self.input_conv(x)
        if g is not None:
            c = c + self.cond(g)

        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)

        total_upsample = int(self.prod_upsample_scales[-1])
        x_sine = self.signal_generator(f0, total_upsample)

        d_list = []
        for i in range(self.num_upsamples):
            f0_layer = torch.nn.functional.interpolate(
                f0, scale_factor=self.prod_upsample_scales[i], mode="nearest"
            )
            d_list.append(
                dilated_factor_torch(f0_layer, self.layer_fs[i], self.dense_factors[i])
            )

        e = c
        x_sine = self.sn["emb"](x_sine)
        sine_embs = [x_sine]
        curr = x_sine
        for i in range(self.num_upsamples - 1):
            curr = self.sn["downsamples"][i](curr)
            sine_embs.append(curr)

        for i in range(self.num_upsamples):
            e = self.sn["upsamples"][i](e)
            e = e + sine_embs[-(i + 1)]
            e = self.sn["blocks"][i](e, d_list[i])

        excitation = self.sn["output_conv"](e)

        embs_filt = [e]
        curr_e = e
        for i in range(self.num_upsamples - 1):
            if self.share_downsamples:
                curr_e = self.sn["downsamples"][i](curr_e)
            else:
                curr_e = self.fn["downsamples"][i](curr_e)
            embs_filt.append(curr_e)

        num_blocks = len(self.filter_network_params["resblock_kernel_sizes"])
        c_feat = c

        for i in range(self.num_upsamples):
            if self.share_upsamples:
                c_feat = self.sn["upsamples"][i](c_feat)
            else:
                c_feat = self.fn["upsamples"][i](c_feat)

            c_feat = c_feat + embs_filt[-(i + 1)]

            for j in range(num_blocks):
                c_feat = self.fn["blocks"][i * num_blocks + j](c_feat)

        waveform = self.fn["output_conv"](c_feat)

        return waveform, excitation

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)
