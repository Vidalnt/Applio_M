import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional, Union, Tuple, List
import numpy as np

#pip install spikingjelly
from spikingjelly.activation_based.neuron import ParametricLIFNode
from spikingjelly.activation_based import functional

class SineGenerator(torch.nn.Module):
    """
    Definition of sine generator

    Generates sine waveforms with optional harmonics and additive noise.
    Can be used to create harmonic noise source for neural vocoders.

    Args:
        samp_rate (int): Sampling rate in Hz.
        harmonic_num (int): Number of harmonic overtones (default 0).
        sine_amp (float): Amplitude of sine-waveform (default 0.1).
        noise_std (float): Standard deviation of Gaussian noise (default 0.003).
        voiced_threshold (float): F0 threshold for voiced/unvoiced classification (default 0).
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: torch.Tensor):
        """
        Generates voiced/unvoiced (UV) signal based on the fundamental frequency (F0).

        Args:
            f0 (torch.Tensor): Fundamental frequency tensor of shape (batch_size, length, 1).
        """
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values: torch.Tensor):
        """
        Generates sine waveforms based on the fundamental frequency (F0) and its harmonics.

        Args:
            f0_values (torch.Tensor): Tensor of fundamental frequency and its harmonics,
                                      shape (batch_size, length, dim), where dim indicates
                                      the fundamental tone and overtones.
        """
        # convert to F0 in rad. The integer part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    def forward(self, f0: torch.Tensor):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """
    Generates harmonic and noise source features.

    This module uses the SineGenerator to create harmonic signals based on the
    fundamental frequency (F0) and merges them into a single excitation signal.

    Args:
        sample_rate (int): Sampling rate in Hz.
        harmonic_num (int, optional): Number of harmonics above F0. Defaults to 0.
        sine_amp (float, optional): Amplitude of sine source signal. Defaults to 0.1.
        add_noise_std (float, optional): Standard deviation of additive Gaussian noise. Defaults to 0.003.
        voiced_threshod (float, optional): Threshold to set voiced/unvoiced given F0. Defaults to 0.
    """

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGenerator(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor):
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        return sine_merge, None, None

class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM) as described in the TS-SNN paper.
    This version uses a fixed shift strategy for increased training stability.
    """
    def __init__(self, channel_folding_factor: int = 32):
        super().__init__()
        self.channel_folding_factor = channel_folding_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [T, B, C, L].
        Returns:
            torch.Tensor: Shifted tensor of the same shape.
        """
        T, B, C, L = x.shape
        
        if T < 2: # Cannot shift if there's only one timestep
            return x

        # C_k in the paper
        num_segments = self.channel_folding_factor
        if C % num_segments != 0:
            raise ValueError(f"Number of channels ({C}) must be divisible by channel_folding_factor ({num_segments}).")
        
        # C_fold in the paper
        fold_size = C // num_segments
        
        # Use fixed split points for stability as requested.
        # A common strategy is to shift 1/4 of channels left, 1/4 right, and keep 1/2 unchanged.
        g1 = num_segments // 4
        g2 = num_segments // 2

        # Initialize a zero tensor for the output
        z = torch.zeros_like(x)

        # Part 1: Shift Left (past information)
        # From group 0 to g1-1
        c_split1 = g1 * fold_size
        if c_split1 > 0:
            z[:-1, :, :c_split1, :] = x[1:, :, :c_split1, :]

        # Part 2: Shift Right (future information)
        # From group g1 to g2-1
        c_split2 = g2 * fold_size
        if c_split2 > c_split1:
            z[1:, :, c_split1:c_split2, :] = x[:-1, :, c_split1:c_split2, :]

        # Part 3: No Shift (present information)
        # From group g2 to the end
        if C > c_split2:
            z[:, :, c_split2:, :] = x[:, :, c_split2:, :]
            
        return z


class MembraneOutputLayer(nn.Module):
    """
    Integrates membrane potentials over time steps to produce a continuous output.
    Assumes input shape [T, B, C, L].
    """
    def __init__(self, timestep: int):
        super().__init__()
        self.n_steps = timestep
        arr = torch.arange(self.n_steps - 1, -1, -1, dtype=torch.float32)
        coef = torch.pow(0.9, arr) # Fixed decay coefficient from original code
        self.register_buffer('coef', coef.view(self.n_steps, 1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Input: [T, B, C, L]
        out = torch.sum(x * self.coef, dim=0) # Output: [B, C, L]
        return out


class SpikingConvNeXtBlock(nn.Module):
    """
    Spiking ConvNeXt Block with TSM and amplitude shortcut.
    Processes inputs over time steps.
    """
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        channel_folding_factor: int = 32,
        tsm_penalty_factor: float = 0.5,
    ):
        super().__init__()
        self.temporal_shift = TemporalShift(channel_folding_factor)
        self.alpha = nn.Parameter(torch.tensor(tsm_penalty_factor))

        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6) # No AdaLayerNorm for simplicity in RVC context
        
        self.neuron1 = ParametricLIFNode(v_threshold=1.0, step_mode='m', backend='torch')
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.neuron2 = ParametricLIFNode(v_threshold=1.0, step_mode='m', backend='torch')
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self,
        potential: torch.Tensor,
        cond_embedding_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, B, C, L = potential.shape
        residual_potential = potential

        # Apply Temporal Shift Module with residual connection
        # The paper's ablation study shows applying TSM in both training and inference is better.
        shifted_potential = self.temporal_shift(potential)
        # Equation (7) from TS-SNN paper: Z' = alpha * Z + X
        x_in = self.alpha * shifted_potential + potential
        
        potential_flat = x_in.flatten(0, 1)

        # Continuous value path, used to generate spikes and amplitude modulation.
        x_continuous = self.dwconv(potential_flat)
        x_continuous = x_continuous.transpose(1, 2)

        # Normalization
        x_continuous = self.norm(x_continuous)

        # Continuous path
        spikes = self.neuron1(x_continuous.reshape(T, B, L, C))
        spikes_flat = spikes.flatten(0, 1)
        h_spikes = self.pwconv1(spikes_flat)
        h_spikes = self.act(h_spikes)
        h_spikes = h_spikes.reshape(T, B, L, -1)
        
        spikes = self.neuron2(h_spikes)
        spikes_flat = spikes.flatten(0,1)
        h_spikes = self.pwconv2(spikes_flat)

        # Injecting amplitude information back
        # Using x_continuous to ensure proper gradient propagation
        modulated_out = h_spikes * x_continuous.clone()

        if self.gamma is not None:
            modulated_out = self.gamma * modulated_out
            
        modulated_out = modulated_out.transpose(1, 2)
        residual_inc = modulated_out.reshape(T, B, C, L)
        
        # Residual connection
        new_potential = residual_potential + residual_inc
        return new_potential, x_in


class SpikingVocosBackbone(nn.Module):
    """
    SpikingVocos backbone adapted for RVC integration.
    Handles gin_channels via a conditioning layer applied before SNN processing.
    """
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        snn_timestep: int,
        layer_scale_init_value: Optional[float] = None,
        gin_channels: int = 0, # Added gin_channels parameter
        channel_folding_factor: int = 32,
        tsm_penalty_factor: float = 0.5,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.snn_timestep = snn_timestep
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        
        # Conditioning layer for gin_channels
        self.gin_channels = gin_channels
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, dim, 1)
        
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList()
        for i in range(num_layers):
            self.convnext.append(
                SpikingConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    channel_folding_factor=channel_folding_factor,
                    tsm_penalty_factor=tsm_penalty_factor,
                )
            )

        self.membrane_output = MembraneOutputLayer(timestep=self.snn_timestep)
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, input_channels, L]

        # Reset SNN states before processing a new batch
        functional.reset_net(self.convnext)
        
        # Input embedding
        x = self.embed(x) # [B, dim, L]

        # Apply conditioning if provided
        if g is not None:
            c = self.cond(g)  # [B, dim, 1] (assuming gin_channels -> dim)
            c = c.expand(-1, -1, x.size(2))  # [B, dim, L] to match x shape
            x = x + c  # Sum conditioning: [B, dim, L] + [B, dim, L] - BILINEAR
        
        # Expand time dimension for SNN simulation
        potential = x.unsqueeze(0).repeat(self.snn_timestep, 1, 1, 1) # [T, B, dim, L]
        
        # Process through SNN backbone
        for conv_block in self.convnext:
            potential, _ = conv_block(potential) # [T, B, dim, L]
        
        # Integrate membrane potentials over time
        x = self.membrane_output(potential) # [B, dim, L]
        
        # Final normalization and transpose
        x = self.final_layer_norm(x.transpose(1, 2)) # [B, L, dim]
        
        return x # Output: [B, L, dim]

class SpikingVocosRVCGenerator(nn.Module):
    """
    Generator wrapper for SpikingVocos adapted for RVC.
    Handles the input/output shapes and gin_channels expected by RVC's Synthesizer.
    """
    def __init__(
        self,
        initial_channel: int, # RVC's inter_channels -> dim for SpikingVocosBackbone
        gin_channels: int,
        snn_timestep: int, # Must be provided
        snn_dim: int = 512, # Hidden dimension for SNN backbone
        snn_intermediate_dim: int = 1536, # Intermediate dimension for SNN blocks
        snn_num_layers: int = 8, # Number of SNN blocks
        sample_rate: int = 40000, # Sample rate for ISTFT
        out_channels: int = 2050, # Output channels for ISTFT (n_fft + 2)
        channel_folding_factor: int = 32,
        tsm_penalty_factor: float = 0.5,
        # ISTFT parameters
        n_fft: int = 2048,
        hop_length: int = 400,
    ):
        super().__init__()
        
        self.snn_timestep = snn_timestep
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Use SpikingVocosBackbone with gin_channels support
        self.backbone = SpikingVocosBackbone(
            input_channels=initial_channel,
            dim=snn_dim,
            intermediate_dim=snn_intermediate_dim,
            num_layers=snn_num_layers,
            snn_timestep=snn_timestep,
            gin_channels=gin_channels, # Pass gin_channels
            channel_folding_factor=channel_folding_factor,
            tsm_penalty_factor=tsm_penalty_factor,
        )

        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num=0)

        # Conv layer to process the harmonic source before concatenation
        self.conv_pre_y = weight_norm(nn.Conv1d(1, snn_dim // 2, kernel_size=7, padding=3))
        # Conv layer to fuse the backbone output (snn_dim) and processed harmonic source (snn_dim//2) -> snn_dim
        self.fuse_y_mel = weight_norm(nn.Conv1d(snn_dim + snn_dim // 2, snn_dim, kernel_size=1))
        
        # Output layer to predict magnitude and phase
        self.out_conv = nn.Conv1d(snn_dim, out_channels, 1)
        
        # ISTFT parameters
        self.istft = torch.istft
        self.window = torch.hann_window(n_fft)

    def forward(self, x: torch.Tensor, f0: Optional[torch.Tensor] = None, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, initial_channel, T_in] where T_in corresponds to time steps of features
        # f0: [B, T_in] fundamental frequency
        # g: [B, gin_channels, 1] global conditioning (speaker embedding)
        
        # Forward through SpikingVocosBackbone
        # Output shape: [B, L, dim] where L is the sequence length (same as input T_in)
        x = self.backbone(x, g=g) # [B, L, dim]
        f0 = None

        if f0 is not None:
            # Generate harmonic source
            # f0 [B, T_in] -> [B, T_in, 1] for m_source
            har_source, _, _ = self.m_source(f0.unsqueeze(-1)) # f0 [B, T_in, 1] -> [B, T_in, 1] (output of m_source)
            # Transpose har_source from [B, T_in, 1] -> [B, 1, T_in] for conv_pre_y
            har_source = har_source.transpose(1, 2) # [B, T_in, 1] -> [B, 1, T_in]
            har_source = self.conv_pre_y(har_source) # [B, 1, T_in] -> [B, snn_dim//2, T_in]
            
            # Concatenate backbone output and harmonic source
            # x is [B, L, snn_dim], har_source is [B, snn_dim//2, T_in]
            # Transpose x to [B, snn_dim, L] for concatenation (assuming L == T_in)
            x_backbone = x.transpose(1, 2) # [B, L, snn_dim] -> [B, snn_dim, T_in]
            x_concat = torch.cat([x_backbone, har_source], dim=1) # [B, snn_dim + snn_dim//2, T_in]
            
            # Fuse concatenated features
            x = self.fuse_y_mel(x_concat) # [B, snn_dim + snn_dim//2, T_in] -> [B, snn_dim, T_in]
            # x is now [B, snn_dim, T_in], ready for out_conv
        
        # If f0 was not provided, x remains [B, L, snn_dim] from backbone, transpose to [B, snn_dim, L]
        else:
            x = x.transpose(1, 2) # [B, L, snn_dim] -> [B, snn_dim, L]
        
        # Predict spectral coefficients: [B, out_channels, L]
        x = self.out_conv(x)
        
        # Split magnitude and phase: [B, out_channels//2, L] each
        mag, phase = x.chunk(2, dim=1)
        
        # Process magnitude and phase
        mag = torch.exp(mag) # Exponentiate magnitude
        mag = torch.clip(mag, max=1e2) # Clip to prevent large values
        phase = phase.float() # Ensure float type
        
        # Reconstruct complex STFT: [B, n_fft//2 + 1, L]
        S = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        
        # Perform ISTFT to get audio: [B, T_audio]
        # Note: The length of the output audio depends on S.shape and hop_length
        audio = self.istft(
            S,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(S.device),
            center=True # Assumes Vocos-style padding was handled upstream
        )
        
        # Add channel dimension for RVC: [B, 1, T_audio]
        audio = audio.unsqueeze(1)
        
        return audio


# SpikingVocosRVCGenerator(
#     initial_channel=192,      # inter_channels
#     gin_channels=256,         # gin_channels
#     snn_timestep=4,          # o 8, debe ser conocido
#     snn_dim=512,             # channel
#     snn_intermediate_dim=1536, # h_channel (o 3 * snn_dim)
#     snn_num_layers=8,        # num_layers
#     out_channels=2050,       # data.filter_length + 2 = 2048 + 2
#     n_fft=2048,              # data.filter_length
#     hop_length=400,          # data.hop_length
#     # channel_folding_factor, tsm_penalty_factor pueden ser los defaults
# )