import math
import torch
from rvc.lib.algorithm.commons import convert_pad_shape, fused_add_tanh_sigmoid_multiply

class MultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention module with optional relative positional encoding and proximal bias.

    Args:
        channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_heads (int): Number of attention heads.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.
        window_size (int, optional): Window size for relative positional encoding. Defaults to None.
        heads_share (bool, optional): Whether to share relative positional embeddings across heads. Defaults to True.
        block_length (int, optional): Block length for local attention. Defaults to None.
        proximal_bias (bool, optional): Whether to use proximal bias in self-attention. Defaults to False.
        proximal_init (bool, optional): Whether to initialize the key projection weights the same as query projection weights. Defaults to False.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int = None,
        heads_share: bool = True,
        block_length: int = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert (
            channels % n_heads == 0
        ), "Channels must be divisible by the number of heads."

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.block_length = block_length
        self.proximal_bias = proximal_bias

        # Define projections
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)

        self.drop = torch.nn.Dropout(p_dropout)

        # Relative positional encodings
        if window_size:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(
                torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = torch.nn.Parameter(
                torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels)
                * rel_stddev
            )

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)
        torch.nn.init.xavier_uniform_(self.conv_o.weight)

        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        # Compute query, key, value projections
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)

        # Compute attention
        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        # Final output projection
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        # Reshape and compute scaled dot-product attention
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        if self.window_size:
            assert t_s == t_t, "Relative attention only supports self-attention."
            scores += self._compute_relative_scores(query, t_s)

        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias only supports self-attention."
            scores += self._attention_bias_proximal(t_s).to(scores.device, scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length:
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)

        # Apply softmax and dropout
        p_attn = self.drop(torch.nn.functional.softmax(scores, dim=-1))

        # Compute attention output
        output = torch.matmul(p_attn, value)

        if self.window_size:
            output += self._apply_relative_values(p_attn, t_s)

        return output.transpose(2, 3).contiguous().view(b, d, t_t), p_attn

    def _compute_relative_scores(self, query, length):
        rel_emb = self._get_relative_embeddings(self.emb_rel_k, length)
        rel_logits = self._matmul_with_relative_keys(
            query / math.sqrt(self.k_channels), rel_emb
        )
        return self._relative_position_to_absolute_position(rel_logits)

    def _apply_relative_values(self, p_attn, length):
        rel_weights = self._absolute_position_to_relative_position(p_attn)
        rel_emb = self._get_relative_embeddings(self.emb_rel_v, length)
        return self._matmul_with_relative_values(rel_weights, rel_emb)

    # Helper methods
    def _matmul_with_relative_values(self, x, y):
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(self, x, y):
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        end = start + 2 * length - 1

        if pad_length > 0:
            embeddings = torch.nn.functional.pad(
                embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        return embeddings[:, start:end]

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]])
        )
        x_flat = x.view(batch, heads, length * 2 * length)
        x_flat = torch.nn.functional.pad(
            x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )
        return x_flat.view(batch, heads, length + 1, 2 * length - 1)[
            :, :, :length, length - 1 :
        ]

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view(batch, heads, length**2 + length * (length - 1))
        x_flat = torch.nn.functional.pad(
            x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]])
        )
        return x_flat.view(batch, heads, length, 2 * length)[:, :, :, 1:]

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = r.unsqueeze(0) - r.unsqueeze(1)
        return -torch.log1p(torch.abs(diff)).unsqueeze(0).unsqueeze(0)


class FFN(torch.nn.Module):
    """
    Feed-forward network module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        filter_channels (int): Number of filter channels in the convolution layers.
        kernel_size (int): Kernel size of the convolution layers.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.
        activation (str, optional): Activation function to use. Defaults to None.
        causal (bool, optional): Whether to use causal padding in the convolution layers. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str = None,
        causal: bool = False,
    ):
        super().__init__()
        self.padding_fn = self._causal_padding if causal else self._same_padding

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = torch.nn.Dropout(p_dropout)

        self.activation = activation

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding_fn(x * x_mask))
        x = self._apply_activation(x)
        x = self.drop(x)
        x = self.conv_2(self.padding_fn(x * x_mask))
        return x * x_mask

    def _apply_activation(self, x):
        if self.activation == "gelu":
            return x * torch.sigmoid(1.702 * x)
        return torch.relu(x)

    def _causal_padding(self, x):
        pad_l, pad_r = self.conv_1.kernel_size[0] - 1, 0
        return torch.nn.functional.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [pad_l, pad_r]])
        )

    def _same_padding(self, x):
        pad = (self.conv_1.kernel_size[0] - 1) // 2
        return torch.nn.functional.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [pad, pad]])
        )

class Depthwise_Separable_Conv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.depth_conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.point_conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
        self.depth_conv = weight_norm(self.depth_conv, name="weight")
        self.point_conv = weight_norm(self.point_conv, name="weight")

    def remove_weight_norm(self):
        self.depth_conv = remove_weight_norm(self.depth_conv, name="weight")
        self.point_conv = remove_weight_norm(self.point_conv, name="weight")


class Depthwise_Separable_TransposeConv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
        dilation=1,
        padding_mode="zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.depth_conv = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            output_padding=output_padding,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.point_conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
        self.depth_conv = weight_norm(self.depth_conv, name="weight")
        self.point_conv = weight_norm(self.point_conv, name="weight")

    def remove_weight_norm(self):
        remove_weight_norm(self.depth_conv, name="weight")
        remove_weight_norm(self.point_conv, name="weight")

def weight_norm_modules(module, name="weight", dim=0):
    if isinstance(module, Depthwise_Separable_Conv1D) or isinstance(
        module, Depthwise_Separable_TransposeConv1D
    ):
        module.weight_norm()
        return module
    else:
        return weight_norm(module, name, dim)


def remove_weight_norm_modules(module, name="weight"):
    if isinstance(module, Depthwise_Separable_Conv1D) or isinstance(
        module, Depthwise_Separable_TransposeConv1D
    ):
        module.remove_weight_norm()
    else:
        remove_weight_norm(module, name)


class FFT(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers=1,
        kernel_size=1,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=True,
        isflow=False,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        if isflow and "gin_channels" in kwargs and kwargs["gin_channels"] > 0:
            cond_layer = torch.nn.Conv1d(
                kwargs["gin_channels"], 2 * hidden_channels * n_layers, 1
            )
            self.cond_pre = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            self.cond_layer = weight_norm_modules(cond_layer, name="weight")
            self.gin_channels = kwargs["gin_channels"]
        self.drop = torch.nn.Dropout(p_dropout)
        self.self_attn_layers = torch.nn.ModuleList()
        self.norm_layers_0 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        """
        x: decoder input
        h: encoder output
        """
        if g is not None:
            g = self.cond_layer(g)

        self_attn_mask = subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )
        x = x * x_mask
        for i in range(self.n_layers):
            if g is not None:
                x = self.cond_pre(x)
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                x = fused_add_tanh_sigmoid_multiply(
                    x, g_l, torch.IntTensor([self.hidden_channels])
                )
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
        x = x * x_mask
        return x