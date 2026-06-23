# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

def propagate(x: torch.Tensor, weight: torch.Tensor, 
              index_kept: torch.Tensor, index_prop: torch.Tensor, 
              standard: str = "None", alpha: Optional[float] = 0, 
              token_scales: Optional[torch.Tensor] = None):
    """
    Propagate tokens based on the selection results.
    ================================================
    Args:
        - x: Tensor([B, N, C]): the feature map of N tokens, including the [CLS] token.
        
        - weight: Tensor([B, N-1, N-1]): the weight of each token propagated to the other tokens, 
                                         excluding the [CLS] token. weight could be a pre-defined 
                                         graph of the current feature map (by default) or the
                                         attention map (need to manually modify the Block Module).
                                         
        - index_kept: Tensor([B, N-1-num_prop]): the index of kept image tokens in the feature map X
        
        - index_prop: Tensor([B, num_prop]): the index of propagated image tokens in the feature map X
        
        - standard: str: the method applied to propagate the tokens, including "None", "Mean" and 
                         "GraphProp"
        
        - alpha: float: the coefficient of propagated features
        
        - token_scales: Tensor([B, N]): the scale of tokens, including the [CLS] token. token_scales
                                        is None by default. If it is not None, then token_scales 
                                        represents the scales of each token and should sum up to N.
        
    Return:
        - x: Tensor([B, N-1-num_prop, C]): the feature map after propagation
        
        - weight: Tensor([B, N-1-num_prop, N-1-num_prop]): the graph of feature map after propagation
        
        - token_scales: Tensor([B, N-1-num_prop]): the scale of tokens after propagation
    """
    
    B, N, C = x.shape
    
    # Step 1: divide tokens
    x_cls = x[:, 0:1] # B, 1, C
    x_kept = x.gather(dim=1, index=index_kept.unsqueeze(-1).expand(-1,-1,C)) # B, N-1-num_prop, C
    x_prop = x.gather(dim=1, index=index_prop.unsqueeze(-1).expand(-1,-1,C)) # B, num_prop, C
    
    # Step 2: divide token_scales if it is not None
    if token_scales is not None:
        token_scales_cls = token_scales[:, 0:1] # B, 1
        token_scales_kept = token_scales.gather(dim=1, index=index_kept) # B, N-1-num_prop
        token_scales_prop = token_scales.gather(dim=1, index=index_prop) # B, num_prop
    
    # Step 3: propagate tokens
    if standard == "None":
        """
        No further propagation
        """
        pass
        
    elif standard == "Mean":
        """
        Calculate the mean of all the propagated tokens,
        and concatenate the result token back to kept tokens.
        """
        # naive average
        x_prop = x_prop.mean(1, keepdim=True) # B, 1, C
        # Concatenate the average token 
        x_kept = torch.cat((x_kept, x_prop), dim=1) # B, N-num_prop, C
            
    elif standard == "GraphProp":
        """
        Propagate all the propagated token to kept token
        with respect to the weights and token scales.
        """
        assert weight is not None, "The graph weight is needed for graph propagation"
        
        # Step 3.1: divide propagation weights.
        index_kept = index_kept - 1 # since weights do not include the [CLS] token
        index_prop = index_prop - 1 # since weights do not include the [CLS] token
        
        weight = weight.gather(dim=1, index=index_kept.unsqueeze(-1).expand(-1,-1,N-1)) # B, N-1-num_prop, N-1
        weight_prop = weight.gather(dim=2, index=index_prop.unsqueeze(1).expand(-1,weight.shape[1],-1)) # B, N-1-num_prop, num_prop
        weight = weight.gather(dim=2, index=index_kept.unsqueeze(1).expand(-1,weight.shape[1],-1)) # B, N-1-num_prop, N-1-num_prop
        
        # Step 3.2: generate the broadcast message and propagate the message to corresponding kept tokens
        # Simple implementation
        x_prop = weight_prop @ x_prop # B, N-1-num_prop, C
        x_kept = x_kept + alpha * x_prop # B, N-1-num_prop, C
        
        """ scatter_reduce implementation for batched inputs
        # Get the non-zero values
        non_zero_indices = torch.nonzero(weight_prop, as_tuple=True)
        non_zero_values = weight_prop[non_zero_indices]
        
        # Sparse multiplication
        batch_indices, row_indices, col_indices = non_zero_indices
        sparse_matmul = alpha * non_zero_values[:, None] * x_prop[batch_indices, col_indices, :]
        reduce_indices = batch_indices * x_kept.shape[1] + row_indices
        
        x_kept = x_kept.reshape(-1, C).scatter_reduce(dim=0, 
                                                      index=reduce_indices[:, None], 
                                                      src=sparse_matmul, 
                                                      reduce="sum",
                                                      include_self=True)
        x_kept = x_kept.reshape(B, -1, C)
        """
        
        # Step 3.3: calculate the scale of each token if token_scales is not None
        if token_scales is not None:
            token_scales_cls = token_scales[:, 0:1] # B, 1
            token_scales = token_scales[:, 1:]
            token_scales_kept = token_scales.gather(dim=1, index=index_kept) # B, N-1-num_prop
            token_scales_prop = token_scales.gather(dim=1, index=index_prop) # B, num_prop
            token_scales_prop = weight_prop @ token_scales_prop.unsqueeze(-1) # B, N-1-num_prop, 1
            token_scales = token_scales_kept + alpha * token_scales_prop.squeeze(-1) # B, N-1-num_prop
            token_scales = torch.cat((token_scales_cls, token_scales), dim=1) # B, N-num_prop
    else:
        assert False, "Propagation method \'%f\' has not been supported yet." % standard
    
    # Step 4： concatenate the [CLS] token and generate returned value
    x = torch.cat((x_cls, x_kept), dim=1) # B, N-num_prop, C
    return x, weight, token_scales



def select(weight: torch.Tensor, standard: str = "None", num_prop: int = 0):
    """
    Select image tokens to be propagated. The [CLS] token will be ignored. 
    ======================================================================
    Args:
        - weight: Tensor([B, H, N, N]): used for selecting the kept tokens. Only support the
                                        attention map of tokens at the moment.
        
        - standard: str: the method applied to select the tokens
        
        - num_prop: int: the number of tokens to be propagated
        
    Return:
        - index_kept: Tensor([B, N-1-num_prop]): the index of kept tokens 
        
        - index_prop: Tensor([B, num_prop]): the index of propagated tokens
    """
    
    assert len(weight.shape) == 4, "Selection methods on tensors other than the attention map haven't been supported yet."
    B, H, N1, N2 = weight.shape
    assert N1 == N2, "Selection methods on tensors other than the attention map haven't been supported yet."
    N = N1
    assert num_prop >= 0, "The number of propagated/pruned tokens must be non-negative."
            
    if standard == "CLSAttnMean":
        token_rank = weight[:,:,0,1:].mean(1)
        
    elif standard == "CLSAttnMax":
        token_rank = weight[:,:,0,1:].max(1)[0]
            
    elif standard == "IMGAttnMean":
        token_rank = weight[:,:,:,1:].sum(-2).mean(1)
    
    elif standard == "IMGAttnMax":
        token_rank = weight[:,:,:,1:].sum(-2).max(1)[0]
            
    elif standard == "DiagAttnMean":
        token_rank = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].mean(1)
        
    elif standard == "DiagAttnMax":
        token_rank = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].max(1)[0]
        
    elif standard == "MixedAttnMean":
        token_rank_1 = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].mean(1)
        token_rank_2 = weight[:,:,:,1:].sum(-2).mean(1)
        token_rank = token_rank_1 * token_rank_2
        
    elif standard == "MixedAttnMax":
        token_rank_1 = torch.diagonal(weight, dim1=-2, dim2=-1)[:,:,1:].max(1)[0]
        token_rank_2 = weight[:,:,:,1:].sum(-2).max(1)[0]
        token_rank = token_rank_1 * token_rank_2
        
    elif standard == "CosSimMean":
        weight = weight[:,:,1:,:].mean(1)
        weight = weight / weight.norm(dim=-1, keepdim=True)
        token_rank = -(weight @ weight.transpose(-1, -2)).sum(-1)
    
    elif standard == "CosSimMax":
        weight = weight[:,:,1:,:].max(1)[0]
        weight = weight / weight.norm(dim=-1, keepdim=True)
        token_rank = -(weight @ weight.transpose(-1, -2)).sum(-1)
        
    elif standard == "Random":
        token_rank = torch.randn((B, N-1), device=weight.device)
            
    else:
        print("Type\'", standard, "\' selection not supported.")
        assert False
        
    token_rank = torch.argsort(token_rank, dim=1, descending=True) # B, N-1
    index_kept = token_rank[:, :-num_prop]+1 # B, N-1-num_prop
    index_prop = token_rank[:, -num_prop:]+1 # B, num_prop
    return index_kept, index_prop
    
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops



class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, selection="None", propagation="None", num_prop=0, sparsity=1, alpha=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process
        
        self.propagation = propagation
        self.selection = selection
        self.num_prop = num_prop
        self.sparsity = sparsity
        self.alpha = alpha

    def forward(self, x, weight, token_scales):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        if self.selection != "None":
            index_kept, index_prop = select(attn, standard=self.selection, num_prop=self.num_prop)
            x, weight, token_scales = propagate(x, weight, index_kept, index_prop, standard=self.propagation,
                                               alpha=self.alpha, token_scales=token_scales)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False, selection="None", propagation="None", num_prop=0, sparsity=1, alpha=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 selection=selection, propagation=propagation, num_prop=num_prop, 
                                 sparsity=sparsity, alpha=alpha)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, graph, token_scales):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, graph, token_scales = checkpoint.checkpoint(blk, x, graph, token_scales)
            else:
                x, graph, token_scales = blk(x, graph, token_scales)
        if self.downsample is not None:
            x, graph, token_scales = self.downsample(x, graph, token_scales)
        return x, graph, token_scales

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,
                selection="None",
                propagation="None",
                num_prop=0,
                num_neighbours=0,
                sparsity=1,
                alpha=0.1,
                token_scale=False,
                graph_type="None",
                pretrained_cfg_overlay=None, **kwargs):
                
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        flag=True
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process,
                               selection=selection,
                               propagation=propagation,
                               num_prop=num_prop,
                               sparsity=sparsity,
                               alpha=alpha)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.token_scale = token_scale
        self.num_neighbours = num_neighbours
        self.graph_type = graph_type
                
        if return_dense:
            self.aux_head=nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
        # First check the graph type is suitable for the propagation method
        if propagation == "GraphProp" and self.graph_type not in ["Spatial", "Semantic", "Mixed"]:
            self.graph_type = "Spatial"
        elif propagation != "GraphProp":
            self.graph_type = "None"
            
        N = (img_size // patch_size)**2
        if self.graph_type in ["Spatial", "Mixed"]:
            # Create a range tensor of node indices
            indices = torch.arange(N)
            # Reshape the indices tensor to create a grid of row and column indices
            row_indices = indices.view(-1, 1).expand(-1, N)
            col_indices = indices.view(1, -1).expand(N, -1)
            # Compute the adjacency matrix
            row1, col1 = row_indices // int(math.sqrt(N)), row_indices % int(math.sqrt(N))
            row2, col2 = col_indices // int(math.sqrt(N)), col_indices % int(math.sqrt(N))
            graph = ((abs(row1 - row2) <= 1).float() * (abs(col1 - col2) <= 1).float())
            graph = graph - torch.eye(N)
            self.spatial_graph = graph.to("cuda") # comment .to("cuda") if the environment is cpu
        
        if self.token_scale:
            self.token_scales = torch.ones([N+1])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        B, N, C = x.shape
        
        if self.graph_type in ["Semantic", "Mixed"]:
            # Generate the semantic graph w.r.t. the cosine similarity between tokens
            # Compute cosine similarity
            x_normed = x[:, 1:] / x[:, 1:].norm(dim=-1, keepdim=True)
            x_cossim = x_normed @ x_normed.transpose(-1, -2)
            threshold = torch.kthvalue(x_cossim, N-1-self.num_neighbours, dim=-1, keepdim=True)[0] # B,H,1,1 
            semantic_graph = torch.where(x_cossim>=threshold, 1.0, 0.0)
            semantic_graph = semantic_graph - torch.eye(N-1, device=semantic_graph.device).unsqueeze(0)
        
        if self.graph_type == "None":
            graph = None
        else:
            if self.graph_type == "Spatial":
                graph = self.spatial_graph.unsqueeze(0).expand(B,-1,-1)#.to(x.device)
            elif self.graph_type == "Semantic":
                graph = semantic_graph
            elif self.graph_type == "Mixed":
                # Integrate the spatial graph and semantic graph
                spatial_graph = self.spatial_graph.unsqueeze(0).expand(B,-1,-1).to(x.device)
                graph = torch.bitwise_or(semantic_graph.int(), spatial_graph.int()).float()
            
            # Symmetrically normalize the graph
            degree = graph.sum(-1) # B, N
            degree = torch.diag_embed(degree**(-1/2))
            graph = degree @ graph @ degree
            
        if self.token_scale:
            token_scales = self.token_scales.unsqueeze(0).expand(B,-1).to(x.device)
        else:
            token_scales = None
            
        for i, layer in enumerate(self.layers):
            x, graph, token_scales  = layer(x, graph, token_scales)
                    

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
        
        
@register_model        
def swin_small(pretrained=False, pretrained_cfg=None, **kwargs):
    model = SwinTransformer(img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=1000,
                            embed_dim=96,
                            depths=[ 2, 2, 18, 2 ],
                            num_heads=[ 3, 6, 12, 24 ],
                            window_size=7,
                            mlp_ratio=4,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=0.1,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False,
                            keep_ratio=0.5, **kwargs)