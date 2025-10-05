# -*- coding: utf-8 -*-

"""QHARMA-GAN Modules adapted for RVC-style upsampling.

This code is based on https://github.com/jik876/hifi-gan
and adapted to match RVC's HiFiGANNSFGenerator upsampling pattern.
"""

import logging
import torch
from rvc.lib.algorithm.residuals import LRELU_SLOPE, ResBlock
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)
import math

from torch.autograd import Function

from torch import Tensor

class AngleFunction(Function):

    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod

    def forward(ctx, x: Tensor):

        ctx.save_for_backward(x)

        return torch.atan2(x.imag, x.real)

    @staticmethod

    def backward(ctx, grad: Tensor):

        (x,) = ctx.saved_tensors

        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-6)

        return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))

def angle(input):
    return AngleFunction.apply(input)

def freq2spec(freqs,fs, n_harm=128, a=None, nfft=1024, sigma = 0.005, out_dim = 3):
    b,t,c = freqs.size()
    
    if freqs.shape[-1] < 3:
        freqs = (freqs) * torch.arange(1, n_harm + 1).to(freqs)
    else:
        freqs = (freqs)    
    mask = ((freqs < (fs/2-100)) & (freqs>0))
    if a is not None:
        if a.shape[-1]> n_harm:
            coef_a = a[...,:int(c/2)] + 1j * a[...,int(c/2):] 
            amp, phase = coef_a.abs()*mask, torch.angle(coef_a)
        elif a.shape[-1] == n_harm:
            amp = a*mask
    else:
        amp = torch.ones_like(freqs)*mask
    # mask = ((freqs < (fs/2-100)) & torch.ge(freqs,0))
    
    amp, freqs = amp.unsqueeze(-1), freqs.unsqueeze(-1)
    w_win = torch.linspace(0,fs/2,int(nfft/2))
    w_idx = w_win.to(freqs) - freqs
    spec = amp*(4*math.pi*sigma**2)**(0.25)*torch.exp(-(w_idx*sigma*2*math.pi)**2/2)*fs/nfft*2
    if out_dim == 3:
        spec = spec.sum(-2)
    return spec
    
class QHARMA_GANGenerator(torch.nn.Module):
    """QHARMA-GAN generator module adapted for RVC-style upsampling."""

    def __init__(
        self,
        in_channels=80,
        out_channels=128, # This is n_harmonic, not final output channels
        channels=512, # Initial internal channels
        kernel_size=7,
        upsample_rates=[10, 10, 2, 2], # RVC-style upsampling rates
        upsample_kernel_sizes=[16, 16, 4, 4], # RVC-style kernel sizes for transpose convs
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        hop=128, # Raw hop length before upsample_rate adjustment
        sampling_rate=22050,
        DAP_order = 128,
        r = 8,
        real = True,
        upsample_rate = 4, # Rate for f0/vuv interpolation
    ):
        """Initialize QHARMA_GANGenerator module with RVC-style upsampling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of harmonic components (n_harmonic).
            channels (int): Number of initial internal channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_rates (list): List of upsampling factors.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers (ConvTranspose1d).
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            hop (int): Hop length for internal calculations.
            sampling_rate (int): Sampling rate.
            DAP_order (int): Order for DAP model.
            r (int): Factor related to DAP order.
            real (bool): Whether to use real AR/MA coefficients.
            upsample_rate (int): Rate for f0/vuv upsampling.

        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_rates) == len(upsample_kernel_sizes), "Length of upsample_rates must match upsample_kernel_sizes."
        assert len(resblock_dilations) == len(resblock_kernel_sizes), "Length of resblock_dilations must match resblock_kernel_sizes."

        self.upsample_rate = upsample_rate
        self.real = real
        # Adjust hop based on upsample_rate as in original code
        self.hop = hop // upsample_rate
        self.fs = sampling_rate
        if DAP_order is None:
            self.DAP_order = int(self.fs/1000) + 2
        else:
            self.DAP_order = DAP_order
        self.nfft = 1024
        self.n_harmonic = out_channels # This is the number of harmonics to generate
        # define modules
        self.num_upsamples = len(upsample_rates) # Number of upsampling stages
        self.num_blocks = len(resblock_kernel_sizes) # Number of residual block types

        # Initial conv layer
        self.input_conv = torch.nn.Conv1d(
                in_channels,
                channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            )
        
        # Calculate intermediate channel sizes for upsampling layers
        self.upsample_initial_channel = channels # Store initial channel count
        self.channels_upsample = [
            self.upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]
        
        # Define upsampling layers 
        self.upsamples = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Calculate input and output channels for this upsampling layer
            in_ch = self.upsample_initial_channel if i == 0 else self.channels_upsample[i-1]
            out_ch = self.channels_upsample[i]
            # Calculate padding for ConvTranspose1d
            # Handling odd upsampling rates as in HiFiGANNSFGenerator
            if u % 2 == 0:
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2

            self.upsamples.append(
                torch.nn.Sequential(
                    torch.nn.LeakyReLU(LRELU_SLOPE),
                    torch.nn.ConvTranspose1d(
                        in_ch, # Input channels
                        out_ch, # Output channels
                        k, # Kernel size
                        u, # Stride (upsampling factor)
                        padding=padding,
                        output_padding=u % 2,
                        bias=bias,
                    ),
                )
            )

        # Define residual blocks
        self.blocks = torch.nn.ModuleList()
        for i in range(len(self.upsamples)): # For each upsampling stage
             for k, d in zip(resblock_kernel_sizes, resblock_dilations): # For each block type
                 # The ResBlock takes the output channels of the current upsampling layer as input
                 self.blocks.append(ResBlock(kernel_size=k, channels=self.channels_upsample[i], dilations=d))

        # Output layers - these now take the final upsampled channel count as input
        final_upsampled_channels = self.channels_upsample[-1]
        self.output_conv = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                final_upsampled_channels, # Input is from last upsampling layer
                out_channels, # Output is n_harmonic
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.LeakyReLU(),
        )

        # AR/MA layers - these also take the final upsampled channel count as input
        if self.real:
            self.AR_A = torch.nn.Sequential(
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv1d(
                        final_upsampled_channels, # Input is from last upsampling layer
                        self.DAP_order,
                        kernel_size,
                        bias=bias,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.Tanh(),
                )
            self.MA_A = torch.nn.Sequential(
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv1d(
                        final_upsampled_channels, # Input is from last upsampling layer
                        self.DAP_order,
                        kernel_size,
                        bias=bias,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.Tanh(),
                )
        else:
            self.AR_A = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    final_upsampled_channels, # Input is from last upsampling layer
                    self.DAP_order,
                    kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.LeakyReLU(),
            )
        self.MA_A = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    final_upsampled_channels, # Input is from last upsampling layer
                    self.DAP_order,
                    kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.LeakyReLU(),
            )
        self.AR_P = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    final_upsampled_channels, # Input is from last upsampling layer
                    self.DAP_order,
                    kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Tanh(),
            )
        self.MA_P = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    final_upsampled_channels, # Input is from last upsampling layer
                    self.DAP_order,
                    kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Tanh(),
            )
        # Gain layer - also takes final upsampled channel count
        self.Gain = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    final_upsampled_channels, # Input is from last upsampling layer
                    1, # Output is 1 (gain per timestep/frame)
                    kernel_size,
                    bias=bias,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.LeakyReLU(),
            )
        self.r = r//2
        
        # apply weight norm
        self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c, p, vuv):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        # network
        p = torch.nn.functional.interpolate(p, size=p.shape[-1]*self.upsample_rate, mode = "linear", align_corners = True)
        vuv = torch.nn.functional.interpolate(vuv, size=vuv.shape[-1]*self.upsample_rate, mode = "linear", align_corners = True)
        x = self.input_conv(c) # x shape: [B, channels, T_up]

        # Process through upsampling and residual blocks
        for i, ups in enumerate(self.upsamples):
            x = ups(x) # x shape: [B, channels_upsample[i], T_up * upsample_rate_cumulative]
            # Apply residual blocks corresponding to this upsampling stage
            # Blocks for stage i are from index i * self.num_blocks to (i + 1) * self.num_blocks - 1
            xs = 0.0  # initialize sum for residual blocks of this stage
            for j in range(self.num_blocks):
                block_index = i * self.num_blocks + j
                xs += self.blocks[block_index](x)
            x = xs / self.num_blocks # Average the outputs of the residual blocks for this stage

        # At this point, x has shape [B, channels_upsample[-1], final_length]
        # Now apply the final processing layers that depend on this final channel size
        Gain = self.Gain(x).permute(0,2,1) # [B, 1, final_length] -> [B, final_length, 1]
        if self.real:
            A_coef = self.AR_A(x).permute(0,2,1) # [B, DAP_order, final_length] -> [B, final_length, DAP_order]
            B_coef = self.MA_A(x).permute(0,2,1) # [B, DAP_order, final_length] -> [B, final_length, DAP_order]
        else:
            A_coef = torch.polar(self.AR_A(x),self.AR_P(x)*math.pi).permute(0,2,1) # [B, final_length, DAP_order_complex]
            B_coef = torch.polar(self.MA_A(x),self.MA_P(x)*math.pi).permute(0,2,1) # [B, final_length, DAP_order_complex]

        # synthesis (This part remains largely the same, assuming it operates correctly on the final length)
        # amplitude
        B, T, _ = p.permute(0,2,1).size() # T is the length after f0/vuv upsampling
        sig_len = T * self.hop # Target signal length

        if p.permute(0,2,1).shape[-1] < 3:
            freq_new = (p.permute(0,2,1)) * torch.arange(1, self.n_harmonic + 1).to(p)
        else:
            freq_new = (p.permute(0,2,1))

        mask = ((freq_new < (self.fs/2-100)) & (freq_new>0))
        order = int(self.DAP_order/self.r)
        wm = 2*math.pi*(freq_new)/self.fs
        Fourier_basis = torch.exp(-1j*wm.unsqueeze(-1)*torch.arange(1, order+1).to(freq_new)) #* mask.unsqueeze(-1).repeat(1,1,1,p_order)
        # Gain, A_coef, B_coef= DAP_a[...,:1], DAP_a[...,1:self.DAP_order+1], DAP_a[...,(self.DAP_order+1):]
        mag_frame, phase_delta = 1, 0
        for i in range(int(self.r)):
            A_func = (1 + torch.bmm(Fourier_basis.reshape(B*(T),self.n_harmonic,order),A_coef[...,i*order:(i+1)*order].to(Fourier_basis.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
            B_func = (1 + torch.bmm(Fourier_basis.reshape(B*(T),self.n_harmonic,order),B_coef[...,i*order:(i+1)*order].to(Fourier_basis.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
            mag_frame *= (B_func.abs()/A_func.abs().clamp_min_(1e-7))
            phase_delta += (angle(B_func) - angle(A_func))
        mag_frame *= Gain

        mag_frame = torch.where(mag_frame.abs()>1, torch.full_like(mag_frame, 0), mag_frame) * mask

        mag = torch.nn.functional.interpolate(mag_frame.permute(0, 2, 1), size=sig_len, mode = "linear", align_corners = True).permute(0, 2, 1)

        freq_new = torch.cat((freq_new, freq_new[:, -1, :].unsqueeze(1)),1)
        phase_delta = torch.cat((phase_delta, phase_delta[:, -1, :].unsqueeze(1)),1)
        # phase difference
        f_mid = (freq_new[:,1:] + freq_new[:,:-1]) / 2
        delta_phase = 2 * math.pi * f_mid * self.hop / self.fs
        phase_frame = phase_delta
        phase_frame[:,1:] += torch.cumsum(delta_phase,dim=1)

        # phase
        frames_idx = torch.arange(0, T + 1).to(p) * self.hop
        frames_idx[-1] =  frames_idx[-1] - 1
        t_idx = torch.arange(0, T * self.hop).to(p)

        coeffs = natural_cubic_spline_coeffs(frames_idx.float(), phase_frame)
        spline = NaturalCubicSpline(coeffs)
        phase = spline.evaluate(t_idx.long())

        wav = 2 * mag *torch.cos(phase)

        return wav.sum(dim = -1).unsqueeze(1), p.permute(0,2,1) # [B, 1, sig_len], [B, T, n_harmonic]

    def decode(self, c, p, vuv, p_scale):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        # network
        p = torch.nn.functional.interpolate(p, size=p.shape[-1]*self.upsample_rate, mode = "linear", align_corners = True)
        vuv = torch.nn.functional.interpolate(vuv, size=vuv.shape[-1]*self.upsample_rate, mode = "linear", align_corners = True)
        x = self.input_conv(c)
        for i, ups in enumerate(self.upsamples):
            x = ups(x)
            xs = 0.0  # initialize
            for j in range(self.num_blocks):
                block_index = i * self.num_blocks + j
                xs += self.blocks[block_index](x)
            x = xs / self.num_blocks
        
        Gain = self.Gain(x).permute(0,2,1)
        if self.real:
            A_coef = self.AR_A(x).permute(0,2,1)
            B_coef = self.MA_A(x).permute(0,2,1)
        else:
            A_coef = torch.polar(self.AR_A(x),self.AR_P(x)*math.pi).permute(0,2,1)
            B_coef = torch.polar(self.MA_A(x),self.MA_P(x)*math.pi).permute(0,2,1)

        # synthesis

        # amplitude 
        B, T, _ = p.permute(0,2,1).size()
        sig_len = T * self.hop

        if p.permute(0,2,1).shape[-1] < 3:
            freq_new = (p.permute(0,2,1)) * torch.arange(1, self.n_harmonic + 1).to(p)
        else:
            freq_new = (p.permute(0,2,1))
        
        freq_new2 = freq_new.clone() * p_scale
        
        # unvoiced 
        mask_uv = ((freq_new < (self.fs/2-100)) & (freq_new>0))
        order = int(self.DAP_order/self.r)
        wm_uv = 2*math.pi*(freq_new)/self.fs 
        Fourier_basis_uv = torch.exp(-1j*wm_uv.unsqueeze(-1)*torch.arange(1, order+1).to(freq_new)) #* mask.unsqueeze(-1).repeat(1,1,1,p_order)
        
        mag_frame_uv, phase_delta_uv = 1, 0
        for i in range(int(self.r)):
            A_func_uv = (1 + torch.bmm(Fourier_basis_uv.reshape(B*(T),self.n_harmonic,order),A_coef[...,i*order:(i+1)*order].to(Fourier_basis_uv.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
            B_func_uv = (1 + torch.bmm(Fourier_basis_uv.reshape(B*(T),self.n_harmonic,order),B_coef[...,i*order:(i+1)*order].to(Fourier_basis_uv.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
            mag_frame_uv *= (B_func_uv.abs()/A_func_uv.abs().clamp_min_(1e-7))
            phase_delta_uv += (angle(B_func_uv) - angle(A_func_uv))
        mag_frame_uv *= Gain
        mag_frame_uv = torch.where(torch.isinf(mag_frame_uv), torch.full_like(mag_frame_uv, 0), mag_frame_uv) * mask_uv * (1 - vuv).permute(0,2,1)
        
        mag_uv = torch.nn.functional.interpolate(mag_frame_uv.permute(0, 2, 1), size=sig_len, mode = "linear", align_corners = True).permute(0, 2, 1)
        
        freq_new = torch.cat((freq_new, freq_new[:, -1, :].unsqueeze(1)),1) 
        phase_delta_uv = torch.cat((phase_delta_uv, phase_delta_uv[:, -1, :].unsqueeze(1)),1) 
        # phase difference
        f_mid_uv = (freq_new[:,1:] + freq_new[:,:-1]) / 2
        delta_phase_uv = 2 * math.pi * f_mid_uv * self.hop / self.fs
        phase_frame_uv = phase_delta_uv
        phase_frame_uv[:,1:] += torch.cumsum(delta_phase_uv,dim=1)
        # phase_frame = torch.cumsum(phase_frame,dim=1)

        # phase
        frames_idx = torch.arange(0, T + 1).to(p) * self.hop
        frames_idx[-1] =  frames_idx[-1] - 1
        t_idx = torch.arange(0, T * self.hop).to(p)
        
        coeffs_uv = natural_cubic_spline_coeffs(frames_idx.float(), phase_frame_uv)
        spline_uv = NaturalCubicSpline(coeffs_uv)
        phase_uv = spline_uv.evaluate(t_idx.long())

        uv = 2 * mag_uv *torch.cos(phase_uv)

        # voiced
        mask = ((freq_new2 < (self.fs/2-100)) & (freq_new2>0))
        wm = 2*math.pi*(freq_new2)/self.fs 
        Fourier_basis = torch.exp(-1j*wm.unsqueeze(-1)*torch.arange(1, order+1).to(freq_new2)) #* mask.unsqueeze(-1).repeat(1,1,1,p_order)
        # Gain, A_coef, B_coef= DAP_a[...,:1], DAP_a[...,1:self.DAP_order+1], DAP_a[...,(self.DAP_order+1):]
        mag_frame, phase_delta = 1, 0
        for i in range(int(self.r)):
            A_func = (1 + torch.bmm(Fourier_basis.reshape(B*(T),self.n_harmonic,order),A_coef[...,i*order:(i+1)*order].to(Fourier_basis.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
            B_func = (1 + torch.bmm(Fourier_basis.reshape(B*(T),self.n_harmonic,order),B_coef[...,i*order:(i+1)*order].to(Fourier_basis.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
            mag_frame *= (B_func.abs()/A_func.abs().clamp_min_(1e-7))
            phase_delta += (angle(B_func) - angle(A_func))
        mag_frame *= Gain
        if p_scale > 1:
            mag_frame *= p_scale
        
        mag_frame = torch.where(torch.isinf(mag_frame), torch.full_like(mag_frame, 0), mag_frame) * mask * vuv.permute(0,2,1)
        
        mag = torch.nn.functional.interpolate(mag_frame.permute(0, 2, 1), size=sig_len, mode = "linear", align_corners = True).permute(0, 2, 1)
        
        freq_new2 = torch.cat((freq_new2, freq_new2[:, -1, :].unsqueeze(1)),1) 
        phase_delta = torch.cat((phase_delta, phase_delta[:, -1, :].unsqueeze(1)),1) 
        # phase difference
        f_mid = (freq_new2[:,1:] + freq_new2[:,:-1]) / 2
        delta_phase = 2 * math.pi * f_mid * self.hop / self.fs
        phase_frame = phase_delta
        phase_frame[:,1:] += torch.cumsum(delta_phase,dim=1)
        # phase_frame = torch.cumsum(phase_frame,dim=1)

        # phase
        coeffs = natural_cubic_spline_coeffs(frames_idx.float(), phase_frame)
        spline = NaturalCubicSpline(coeffs)
        phase = spline.evaluate(t_idx.long())

        v = 2 * mag *torch.cos(phase)
        wav = uv + v

        return wav.sum(dim = -1).unsqueeze(1), p.permute(0,2,1)
    
    def analysis(self,c,*args):
        # network
        x = self.input_conv(c)
        for i, ups in enumerate(self.upsamples):
            x = ups(x)
            xs = 0.0  # initialize
            for j in range(self.num_blocks):
                block_index = i * self.num_blocks + j
                xs += self.blocks[block_index](x)
            x = xs / self.num_blocks
        
        Gain = self.Gain(x).permute(0,2,1)
        if self.real:
            A_coef = self.AR_A(x).permute(0,2,1)
            B_coef = self.MA_A(x).permute(0,2,1)
        else:
            A_coef = torch.polar(self.AR_A(x),self.AR_P(x)*math.pi).permute(0,2,1)
            B_coef = torch.polar(self.MA_A(x),self.MA_P(x)*math.pi).permute(0,2,1)

        return Gain, A_coef, B_coef

    def synthesis(self, *args):
        if len(args) == 5:
            Gain, A_coef, B_coef, p, vuv = args
            p_scale = 1  # default
        elif len(args) == 6:
            Gain, A_coef, B_coef, p, vuv, p_scale = args
        else:
            raise ValueError("analysis expect 5 or 6 parameters")
        
        p = torch.nn.functional.interpolate(p, size=p.shape[-1]*self.upsample_rate, mode = "linear", align_corners = True)
        # synthesis

        # amplitude 
        B, T, _ = p.permute(0,2,1).size()
        sig_len = T * self.hop

        if p.permute(0,2,1).shape[-1] < 3:
            freq_new = (p.permute(0,2,1)) * torch.arange(1, self.n_harmonic + 1).to(p)
        else:
            freq_new = (p.permute(0,2,1))
        
        if p_scale == 1:
        
            mask = ((freq_new < (self.fs/2-100)) & (freq_new>0))
            order = int(self.DAP_order/self.r)
            wm = 2*math.pi*(freq_new)/self.fs 
            Fourier_basis = torch.exp(-1j*wm.unsqueeze(-1)*torch.arange(1, order+1).to(freq_new)) #* mask.unsqueeze(-1).repeat(1,1,1,p_order)
            mag_frame, phase_delta = 1, 0
            for i in range(int(self.r)):
                A_func = (1 + torch.bmm(Fourier_basis.reshape(B*(T),self.n_harmonic,order),A_coef[...,i*order:(i+1)*order].to(Fourier_basis.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
                B_func = (1 + torch.bmm(Fourier_basis.reshape(B*(T),self.n_harmonic,order),B_coef[...,i*order:(i+1)*order].to(Fourier_basis.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
                mag_frame *= (B_func.abs()/A_func.abs().clamp_min_(1e-7))
                phase_delta += (angle(B_func) - angle(A_func))
            mag_frame *= Gain
            mag_frame = torch.where(mag_frame.abs()>1, torch.full_like(mag_frame, 0), mag_frame) * mask
            mag = torch.nn.functional.interpolate(mag_frame.permute(0, 2, 1), size=sig_len, mode = "linear", align_corners = True).permute(0, 2, 1)
            
            freq_new = torch.cat((freq_new, freq_new[:, -1, :].unsqueeze(1)),1) 
            phase_delta = torch.cat((phase_delta, phase_delta[:, -1, :].unsqueeze(1)),1) 
            # phase difference
            f_mid = (freq_new[:,1:] + freq_new[:,:-1]) / 2
            delta_phase = 2 * math.pi * f_mid * self.hop / self.fs
            phase_frame = phase_delta
            phase_frame[:,1:] += torch.cumsum(delta_phase,dim=1)
            # phase_frame = torch.cumsum(phase_frame,dim=1)

            # phase
            frames_idx = torch.arange(0, T + 1).to(p) * self.hop
            frames_idx[-1] =  frames_idx[-1] - 1
            t_idx = torch.arange(0, T * self.hop).to(p)
            
            coeffs = natural_cubic_spline_coeffs(frames_idx.float(), phase_frame)
            spline = NaturalCubicSpline(coeffs)
            phase = spline.evaluate(t_idx.long())

            wav = 2 * mag *torch.cos(phase)
        else:
            vuv = torch.nn.functional.interpolate(vuv, size=vuv.shape[-1]*self.upsample_rate, mode = "linear", align_corners = True)
            freq_new2 = freq_new.clone() * p_scale
        
            # unvoiced 
            mask_uv = ((freq_new < (self.fs/2-100)) & (freq_new>0))
            order = int(self.DAP_order/self.r)
            wm_uv = 2*math.pi*(freq_new)/self.fs 
            Fourier_basis_uv = torch.exp(-1j*wm_uv.unsqueeze(-1)*torch.arange(1, order+1).to(freq_new)) #* mask.unsqueeze(-1).repeat(1,1,1,p_order)
            # Gain, A_coef, B_coef= DAP_a[...,:1], DAP_a[...,1:self.DAP_order+1], DAP_a[...,(self.DAP_order+1):]
            mag_frame_uv, phase_delta_uv = 1, 0
            for i in range(int(self.r)):
                A_func_uv = (1 + torch.bmm(Fourier_basis_uv.reshape(B*(T),self.n_harmonic,order),A_coef[...,i*order:(i+1)*order].to(Fourier_basis_uv.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
                B_func_uv = (1 + torch.bmm(Fourier_basis_uv.reshape(B*(T),self.n_harmonic,order),B_coef[...,i*order:(i+1)*order].to(Fourier_basis_uv.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
                mag_frame_uv *= (B_func_uv.abs()/A_func_uv.abs().clamp_min_(1e-7))
                phase_delta_uv += (angle(B_func_uv) - angle(A_func_uv))
            mag_frame_uv *= Gain
            # mag_frame, phase_delta = Gain * B_func.abs()/A_func.abs().clamp_min_(1e-7), (torch.angle(B_func) - torch.angle(A_func)) * 4 
            mag_frame_uv = torch.where(torch.isinf(mag_frame_uv), torch.full_like(mag_frame_uv, 0), mag_frame_uv) * mask_uv * (1 - vuv).permute(0,2,1)
            # phase_delta = torch.where(phase_delta.abs() > math.pi, torch.full_like(phase_delta, 1), phase_delta)

            mag_uv = torch.nn.functional.interpolate(mag_frame_uv.permute(0, 2, 1), size=sig_len, mode = "linear", align_corners = True).permute(0, 2, 1)
            
            freq_new = torch.cat((freq_new, freq_new[:, -1, :].unsqueeze(1)),1) 
            phase_delta_uv = torch.cat((phase_delta_uv, phase_delta_uv[:, -1, :].unsqueeze(1)),1) 
            # phase difference
            f_mid_uv = (freq_new[:,1:] + freq_new[:,:-1]) / 2
            delta_phase_uv = 2 * math.pi * f_mid_uv * self.hop / self.fs
            phase_frame_uv = phase_delta_uv
            phase_frame_uv[:,1:] += torch.cumsum(delta_phase_uv,dim=1)
            # phase_frame = torch.cumsum(phase_frame,dim=1)

            # phase
            frames_idx = torch.arange(0, T + 1).to(p) * self.hop
            frames_idx[-1] =  frames_idx[-1] - 1
            t_idx = torch.arange(0, T * self.hop).to(p)
            
            coeffs_uv = natural_cubic_spline_coeffs(frames_idx.float(), phase_frame_uv)
            spline_uv = NaturalCubicSpline(coeffs_uv)
            phase_uv = spline_uv.evaluate(t_idx.long())

            uv = 2 * mag_uv *torch.cos(phase_uv)

            # voiced
            mask = ((freq_new2 < (self.fs/2-100)) & (freq_new2>0))
            wm = 2*math.pi*(freq_new2)/self.fs 
            Fourier_basis = torch.exp(-1j*wm.unsqueeze(-1)*torch.arange(1, order+1).to(freq_new2)) #* mask.unsqueeze(-1).repeat(1,1,1,p_order)
            # Gain, A_coef, B_coef= DAP_a[...,:1], DAP_a[...,1:self.DAP_order+1], DAP_a[...,(self.DAP_order+1):]
            mag_frame, phase_delta = 1, 0
            for i in range(int(self.r)):
                A_func = (1 + torch.bmm(Fourier_basis.reshape(B*(T),self.n_harmonic,order),A_coef[...,i*order:(i+1)*order].to(Fourier_basis.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
                B_func = (1 + torch.bmm(Fourier_basis.reshape(B*(T),self.n_harmonic,order),B_coef[...,i*order:(i+1)*order].to(Fourier_basis.dtype).reshape(B*(T),order,1))).reshape(B,(T),self.n_harmonic)
                mag_frame *= (B_func.abs()/A_func.abs().clamp_min_(1e-7))
                phase_delta += (angle(B_func) - angle(A_func))
            mag_frame *= Gain
            # mag_frame, phase_delta = Gain * B_func.abs()/A_func.abs().clamp_min_(1e-7), (torch.angle(B_func) - torch.angle(A_func)) * 4 
            mag_frame = torch.where(torch.isinf(mag_frame), torch.full_like(mag_frame, 0), mag_frame) * mask * vuv.permute(0,2,1)
            # phase_delta = torch.where(phase_delta.abs() > math.pi, torch.full_like(phase_delta, 1), phase_delta)

            mag = torch.nn.functional.interpolate(mag_frame.permute(0, 2, 1), size=sig_len, mode = "linear", align_corners = True).permute(0, 2, 1)
            
            freq_new2 = torch.cat((freq_new2, freq_new2[:, -1, :].unsqueeze(1)),1) 
            phase_delta = torch.cat((phase_delta, phase_delta[:, -1, :].unsqueeze(1)),1) 
            # phase difference
            f_mid = (freq_new2[:,1:] + freq_new2[:,:-1]) / 2
            delta_phase = 2 * math.pi * f_mid * self.hop / self.fs
            phase_frame = phase_delta
            phase_frame[:,1:] += torch.cumsum(delta_phase,dim=1)
            # phase_frame = torch.cumsum(phase_frame,dim=1)

            # phase
            coeffs = natural_cubic_spline_coeffs(frames_idx.float(), phase_frame)
            spline = NaturalCubicSpline(coeffs)
            phase = spline.evaluate(t_idx.long())

            v = 2 * mag *torch.cos(phase)
            wav = uv + v
        return wav.sum(dim = -1).unsqueeze(1)
    
    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                 _remove_weight_norm(module)
        # Assuming ResBlocks also have a remove_weight_norm method:
        for block in self.blocks:
            if hasattr(block, 'remove_weight_norm'):
                block.remove_weight_norm()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                 _apply_weight_norm(module)

    def inference(self, c, f0, vuv, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        # if not isinstance(c, torch.Tensor):
        #     c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        # if not isinstance(c, torch.Tensor):
        #     f0 = torch.tensor(f0, dtype=torch.float).to(next(self.parameters()).device)    
        # if normalize_before:
        #     c = (c - self.mean) / self.scale
        wav, f = self.forward(c.transpose(1, 0).unsqueeze(0),f0.transpose(1, 0).unsqueeze(0),vuv.transpose(1, 0).unsqueeze(0))
        return wav.squeeze(0).transpose(1, 0), f.squeeze(0)