import math
from typing import Optional, List, Tuple

import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights
from rvc.lib.algorithm.generators.hifigan_nsf import SourceModuleHnNSF
from rvc.lib.algorithm.residuals import apply_mask, ResBlock
from timm.models.layers import DropPath

class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        # x.shape = (B, C, T) -> transpose to (B, T, C) for F.layer_norm
        return torch.nn.functional.layer_norm(x.transpose(1, 2), self.normalized_shape, self.weight, self.bias, self.eps).transpose(1, 2)

class ConvNeXtBlock(torch.nn.Module):
    """
    ConvNeXt block as described in the paper for the Context Aware Module (CAM).
    Reference: Figure 1 and Section 4.3 of the paper.
    """
    def __init__(self, dim, kernel_size=7, expansion=4, drop_path=0.):
        super().__init__()
        self.dw_conv = torch.nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm(dim)
        self.pw_conv1 = torch.nn.Conv1d(dim, dim * expansion, kernel_size=1)
        self.act = torch.nn.SiLU()
        self.pw_conv2 = torch.nn.Conv1d(dim * expansion, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

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
    def __init__(self, channels: int, kernel_size: int = 3, dilations: Tuple[int] = (1, 3, 5)):
        super().__init__(channels, kernel_size, dilations)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt_residual = x
            x = torch.nn.functional.silu(x)
            x = apply_mask(x, x_mask)
            x = c1(x)
            x = torch.nn.functional.silu(x)
            x = apply_mask(x, x_mask)
            x = c2(x)
            x = x + xt_residual
        return apply_mask(x, x_mask)


class ContextAwareModule(torch.nn.Module):
    def __init__(self, dims=[128, 256, 384, 512], depths=[3, 3, 9, 3], drop_path_rate=0.2):
        super().__init__()
        
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        self.stages = torch.nn.ModuleList()
        current_dim = dims[0]
        idx = 0
        
        for i, depth in enumerate(depths):
            blocks = []
            for j in range(depth):
                blocks.append(ConvNeXtBlock(
                    dim=current_dim, 
                    drop_path=dpr[idx + j]
                ))
            self.stages.append(torch.nn.Sequential(*blocks))
            
            # Perform dimension transition if necessary
            if i < len(dims) - 1 and dims[i+1] != current_dim:
                self.stages.append(torch.nn.Sequential(
                    LayerNorm(current_dim),
                    nn.Conv1d(current_dim, dims[i+1], kernel_size=1)
                ))
                current_dim = dims[i+1]
            
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

class EvaGanGenerator(torch.nn.Module):
    def __init__(
        self,
        initial_channel: int, #128
        cam_depths: List[int], #[3, 3, 9, 3] 
        cam_dims: List[int], #[128, 256, 384, 512]
        drop_path_rate: float, #0.2
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        checkpointing: bool = False,
    ):
        super(EvaGanGenerator, self).__init__()
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing

        self.conv_pre = torch.nn.Conv1d(
            initial_channel, cam_dims[0], 7, 1, padding=3
        )
        self.norm_pre = LayerNorm(cam_dims[0])

        self.cam = ContextAwareModule(cam_dims, cam_depths, drop_path_rate)

        assert cam_dims[-1] == upsample_initial_channel, \
            f"CAM out dim {cam_dims[-1]} must equal upsample_initial_channel {upsample_initial_channel}"

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # handling odd upsampling rates
            if u % 2 == 0:
                # old method
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2

            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        channels[i],
                        k,
                        u,
                        padding=padding,
                        output_padding=u % 2,
                    )
                )
            )
            """ handling odd upsampling rates
            #  s   k   p
            # 40  80  20
            # 32  64  16
            #  4   8   2
            #  2   3   1
            # 63 125  31
            #  9  17   4
            #  3   5   1
            #  1   1   0
            """

        self.resblocks = torch.nn.ModuleList(
            [
                EvaResBlock(channels[i], k, d)
                for i in range(len(self.ups))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

        self.conv_post = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        x = self.conv_pre(x)
        # Initial normalization for stability
        x = self.norm_pre(x) 

        if self.training and self.checkpointing:
            x = checkpoint(self.cam, x, use_reentrant=False)
        else:
            x = self.cam(x)

        if g is not None:
            x = x + self.cond(g)

        for i, ups in enumerate(self.ups):
            x = torch.nn.functional.silu(x)
            # Apply upsampling layer
            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                xs = sum(
                    [
                        checkpoint(resblock, x, use_reentrant=False)
                        for j, resblock in enumerate(self.resblocks)
                        if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)
                    ]
                )
            else:
                x = ups(x)
                xs = sum(
                    [
                        resblock(x)
                        for j, resblock in enumerate(self.resblocks)
                        if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)
                    ]
                )
            x = xs / self.num_kernels

        x = torch.nn.functional.silu(x)
        x = torch.tanh(self.conv_post(x))

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

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
        return self