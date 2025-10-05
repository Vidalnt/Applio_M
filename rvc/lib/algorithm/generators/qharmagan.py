# -*- coding: utf-8 -*-
"""
QHARMA-GAN Generator adapted for RVC (Retrieval-based Voice Conversion).
This module wraps the QHARMA_GANGenerator_fast to fit RVC's generator interface.
It calculates the required 'vuv' signal internally from the provided 'f0'.
"""

import torch
from rvc.lib.algorithm.generators.modules import QHARMA_GANGenerator


class QHARMA_GAN_RVCGenerator(torch.nn.Module):
    """QHARMA-GAN Generator adapted for RVC usage.
    Designed to receive parameters from RVC's config.json model section.

    Args:
        initial_channel (int): Number of input mel-spectrogram channels (e.g., n_mels, from config 'inter_channels').
        gin_channels (int): Number of conditioning channels (e.g., speaker embedding dim, from config 'gin_channels').
        sample_rate (int): Target sample rate for synthesis (e.g., from config 'data.sample_rate').
        out_channels_qh (int): Number of output channels for the QHARMA model (e.g., from config 'n_harmonic' or similar).
                               Default is often related to harmonic modeling (e.g., 128).
        channels_qh (int): Number of internal channels for the QHARMA model (e.g., from config 'upsample_initial_channel' or similar).
                           Default is often a high value like 512.
        kernel_size_qh (int): Kernel size for initial/final conv layers in QHARMA. Default is 7.
        upsample_scales (list or tuple): Upsampling scales (e.g., from config 'upsample_rates' or calculated).
        upsample_kernel_sizes (list or tuple): Kernel sizes for upsampling layers (e.g., from config 'upsample_kernel_sizes' or derived).
        resblock_kernel_sizes (list): Kernel sizes for residual blocks (e.g., from config 'resblock_kernel_sizes').
        resblock_dilations (list): Dilations for residual blocks (e.g., from config 'resblock_dilation_sizes').
    """

    def __init__(
        self,
        initial_channel,
        gin_channels=0,
        sample_rate=40000,
        out_channels_qh=512,
        channels_qh=512,
        kernel_size_qh=7,
        upsample_rates=list,
        upsample_kernel_sizes=list,
        resblock_kernel_sizes=list,
        resblock_dilations=list,
        DAP_order=256,
        r=16,
        real=False,
        upsample_rate_qh=4,
    ):
        super(QHARMA_GAN_RVCGenerator, self).__init__()
        self.gin_channels = gin_channels

        self.qharmagan_model = QHARMA_GANGenerator(
            in_channels=initial_channel,
            out_channels=out_channels_qh,
            channels=channels_qh,
            kernel_size=kernel_size_qh,
            upsample_scales=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilations,
            hop=sample_rate // 100,
            sampling_rate=sample_rate,
            DAP_order=DAP_order,
            r=r,
            real=real,
            upsample_rate=upsample_rate_qh,
        )

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, initial_channel, 1)

    def forward(self, x, g=None, f0=None):
        """Forward pass for RVC generator.

        Args:
            x (Tensor): Input mel-spectrogram tensor (batch, initial_channel, time).
            g (Tensor, optional): Global conditioning tensor (e.g., speaker embedding) (batch, gin_channels, 1).
            f0 (Tensor, optional): Fundamental frequency tensor (batch, 1, time). Required by QHARMA-GAN.

        Returns:
            Tensor: Generated waveform tensor (batch, 1, length).
        """
        if g is not None:
            x = x + self.cond(g)

        if f0 is None:
            raise ValueError(
                "QHARMA-GAN requires 'f0' to generate 'vuv' signal or 'vuv' must be provided explicitly. 'f0' was not provided."
            )

        vuv = (f0 > 0.0).float()

        # x.shape = [B, n_mels, T]
        # f0.shape = [B, 1, T]
        # vuv.shape = [B, 1, T]
        waveform, f_out = self.qharmagan_model(x, f0, vuv)
        # waveform [B, 1, length]
        return waveform

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            if isinstance(
                m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Linear)
            ):
                try:
                    torch.nn.utils.remove_weight_norm(m)
                except ValueError:
                    pass

        self.qharmagan_model.apply(_remove_weight_norm)
