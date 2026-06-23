import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.init as init
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple
import torch.utils.checkpoint as checkpoint
from typing import Optional
from timm.models._registry import register_model
import timm

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import numpy as np

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'LV_ViT_Tiny': _cfg(),
    'LV_ViT': _cfg(),
    'LV_ViT_Medium': _cfg(crop_pct=1.0),
    'LV_ViT_Large': _cfg(crop_pct=1.0),
}

DROPOUT_FLOPS = 4
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8
SOFTMAX_FLOPS = 5


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
    

class GroupLinear(nn.Module):
    '''
    Group Linear operator 
    '''
    def __init__(self, in_planes, out_channels,groups=1, bias=True):
        super(GroupLinear, self).__init__()
        assert in_planes%groups==0
        assert out_channels%groups==0
        self.in_dim = in_planes
        self.out_dim = out_channels
        self.groups=groups
        self.bias = bias
        self.group_in_dim = int(self.in_dim/self.groups)
        self.group_out_dim = int(self.out_dim/self.groups)

        self.group_weight = nn.Parameter(torch.zeros(self.groups, self.group_in_dim, self.group_out_dim))
        self.group_bias=nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x):
        t,b,d=x.size()
        x = x.view(t,b,self.groups,int(d/self.groups))
        out = torch.einsum('tbgd,gdf->tbgf', (x, self.group_weight)).reshape(t,b,self.out_dim)+self.group_bias
        return out
    def extra_repr(self):
        s = ('{in_dim}, {out_dim}')
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Mlp(nn.Module):
    '''
    MLP with support to use group linear operator
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., group=1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group==1:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = GroupLinear(in_features, hidden_features,group)
            self.fc2 = GroupLinear(hidden_features, out_features,group)
        self.act = act_layer()

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GroupNorm(nn.Module):
    def __init__(self, num_groups, embed_dim, eps=1e-5, affine=True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, embed_dim,eps,affine)

    def forward(self, x):
        B,T,C = x.shape
        x = x.view(B*T,C)
        x = self.gn(x)
        x = x.view(B,T,C)
        return x


class Attention(nn.Module):
    '''
    Multi-head self-attention
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with some modification to support different num_heads and head_dim.
    '''
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sparsity=1):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim=head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, self.head_dim* self.num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim* self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sparsity = sparsity
        
    def forward(self, x, token_scales=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if token_scales is not None:
            attn = attn + token_scales.log().reshape(B, 1, 1, N)
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.sparsity < 1:
            # Fast implementation for filtering out sparsity% trivial values.
            k = int(N*N*(1-self.sparsity))
            threshold = torch.kthvalue(attn.reshape(B,self.num_heads, -1), k, dim=-1, keepdim=True)[0].unsqueeze(-1) # B,H,1,1
            if self.training:
                # during training, we cannot replace the elements, otherwise it leads to backward propagation errors.
                mask = attn>=threshold
                attn = attn * mask.float()
            else:
                attn[attn<threshold] = 0.0
            
            # Legacy but stable implementation
            # attn_rank = torch.sort(attn.reshape(B,self.num_heads,-1), dim=-1, descending=True)[0]
            # attn_sigma = attn_rank[:,:,int(N*N*self.sparsity)].reshape(B,self.num_heads,1,1).expand(B,self.num_heads,N,N)
            # attn = torch.where(attn>=attn_sigma, attn, 0.0)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
                
        
class Block(nn.Module):
    '''
    Pre-layernorm transformer block
    '''
    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, skip_lam=1., selection="None", propagation="None", num_prop=0, sparsity=1, alpha=0):
        super().__init__()
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.skip_lam = skip_lam

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop, group=group)
        
        self.propagation = propagation
        self.selection = selection
        self.num_prop = num_prop
        self.sparsity = sparsity
        self.alpha = alpha                

    def forward(self, x, weight, token_scales=None):
        tmp, attn = self.attn(self.norm1(x), token_scales)
        x = x + self.drop_path(tmp)/self.skip_lam
        
        if self.selection != "None":
            index_kept, index_prop = select(attn, standard=self.selection, num_prop=self.num_prop)
            x, weight, token_scales = propagate(x, weight, index_kept, index_prop, standard=self.propagation,
                                               alpha=self.alpha, token_scales=token_scales)
                                               
        x = x + self.drop_path(self.mlp(self.norm2(x)))/self.skip_lam
        return x, weight, token_scales
        
        
class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim,kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.proj(x)
        return x


class PatchEmbedNaive(nn.Module):
    """ 
    Image to Patch Embedding
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
        proj=img_size*img_size*3*self.embed_dim,
        )
        return sum(block_flops.values())


class PatchEmbed4_2(nn.Module):
    """ 
    Image to Patch Embedding with 4 layer convolution
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(64)

        self.proj = nn.Conv2d(64, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
        conv1=img_size/2*img_size/2*3*64*7*7,
        conv2=img_size/2*img_size/2*64*64*3*3,
        conv3=img_size/2*img_size/2*64*64*3*3,
        proj=img_size/2*img_size/2*64*self.embed_dim,
        )
        return sum(block_flops.values())

    
class PatchEmbed4_2_128(nn.Module):
    """ 
    Image to Patch Embedding with 4 layer convolution and 128 filters
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, 128, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(128)

        self.proj = nn.Conv2d(128, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x
    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
        conv1=img_size/2*img_size/2*3*128*7*7,
        conv2=img_size/2*img_size/2*128*128*3*3,
        conv3=img_size/2*img_size/2*128*128*3*3,
        proj=img_size/2*img_size/2*128*self.embed_dim,
        )
        return sum(block_flops.values())


def get_block(block_type, **kargs):
    if block_type=='mha':
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type=='ffn':
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type=='tr':
        # transformer block
        return Block(**kargs)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_dpr(drop_path_rate,depth,drop_path_decay='linear'):
    if drop_path_decay=='linear':
        # linear dpr decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    elif drop_path_decay=='fix':
        # use fixed dpr
        dpr= [drop_path_rate]*depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate)==depth
        dpr=drop_path_rate
    return dpr

class IdleLVViT(nn.Module):
    """ Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', hybrid_backbone=None, norm_layer=nn.LayerNorm, p_emb='4_2', head_dim = None, skip_lam = 1.0,order=None, mix_token=False, return_dense=False,
            selection="None",
            propagation="None",
            num_prop=0,
            num_neighbours=0,
            sparsity=1,
            alpha=0.1,
            token_scale=False,
            graph_type="None",
            pretrained_cfg_overlay=None):
                        
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim = embed_dim if num_classes==0 else num_classes
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if p_emb=='4_2':
                patch_embed_fn = PatchEmbed4_2
            elif p_emb=='4_2_128':
                patch_embed_fn = PatchEmbed4_2_128
            else:
                patch_embed_fn = PatchEmbedNaive

            self.patch_embed = patch_embed_fn(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr=get_dpr(drop_path_rate, depth, drop_path_decay)
        self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam,
                selection=selection,
                propagation=propagation,
                num_prop=num_prop,
                sparsity=sparsity,
                alpha=alpha)
                for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.return_dense=return_dense
        self.mix_token=mix_token
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
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
            
        for blk in self.blocks:
            x, graph, token_scales = blk(x, graph, token_scales)
        
        
        x = self.norm(x)
        x_cls = self.head(x[:,0])
        x_aux = self.aux_head(x[:,1:])
        final_pred =  x_cls + 0.5 * x_aux.max(1)[0]

        return final_pred
        

class LVViT(nn.Module):
    """ Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', hybrid_backbone=None, norm_layer=nn.LayerNorm, p_emb='4_2', head_dim = None, skip_lam = 1.0,order=None, mix_token=False, return_dense=False,
            selection="None",
            propagation="None",
            num_prop=0,
            num_neighbours=0,
            sparsity=1,
            alpha=0.1,
            token_scale=False,
            graph_type="None",
            pretrained_cfg_overlay=None):
                        
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim = embed_dim if num_classes==0 else num_classes
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if p_emb=='4_2':
                patch_embed_fn = PatchEmbed4_2
            elif p_emb=='4_2_128':
                patch_embed_fn = PatchEmbed4_2_128
            else:
                patch_embed_fn = PatchEmbedNaive

            self.patch_embed = patch_embed_fn(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr=get_dpr(drop_path_rate, depth, drop_path_decay)
        self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam,
                selection=selection,
                propagation=propagation,
                num_prop=num_prop,
                sparsity=sparsity,
                alpha=alpha)
                for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.return_dense=return_dense
        self.mix_token=mix_token
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        
        x = self.norm(x)
        x_cls = self.head(x[:,0])
        x_aux = self.aux_head(x[:,1:])
        final_pred =  x_cls + 0.5 * x_aux.max(1)[0]

        return final_pred


  
@register_model
def lvvit_s(pretrained=False, pretrained_cfg=None, **kwargs):
    model = IdleLVViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    return model
    
@register_model
def lvvit_m(pretrained=False,  pretrained_cfg=None, **kwargs):
    model = IdleLVViT(patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT_Medium']
    return model