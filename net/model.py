
## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch, torchvision
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time

import clip
import os
import sys
from net.arch_util import LayerNorm2d
from net.local_arch import Local_Base
import torchvision.transforms as transforms
from huggingface_hub import PyTorchModelHubMixin

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))
    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, x):
        x = self.proj(x)
        return x

##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt

class EfficientLargeKernel(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dw_13 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim),
            nn.Conv2d(dim, dim, (1, 3), padding=(0, 2), dilation=2, groups=dim),
            nn.Conv2d(dim, dim, (1, 3), padding=(0, 4), dilation=4, groups=dim),
            nn.Conv2d(dim, dim, (1, 3), padding=(0, 8), dilation=8, groups=dim),
        )  #  1×15
        
        self.dw_31 = nn.Sequential(
            nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim),
            nn.Conv2d(dim, dim, (3, 1), padding=(2, 0), dilation=2, groups=dim),
            nn.Conv2d(dim, dim, (3, 1), padding=(4, 0), dilation=4, groups=dim),
            nn.Conv2d(dim, dim, (3, 1), padding=(8, 0), dilation=8, groups=dim),
        )  # 15×1
        
        self.dw_33 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim),
            nn.Conv2d(dim, dim, 3, padding=4, dilation=4, groups=dim),
            nn.Conv2d(dim, dim, 3, padding=8, dilation=8, groups=dim),
        )  #  15×15

    def forward(self, x):
        return self.dw_13(x) + self.dw_31(x) + self.dw_33(x)

class FDN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU(),
            nn.Conv2d(dim*2, dim, 1),
        )

        self.high_freq_branch = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU(),
            nn.Conv2d(dim*2, dim, 1),
        )

        self.enhancement = nn.Sequential(
            nn.Conv2d(dim*2, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//4, 1),
            nn.GELU(),
            nn.Conv2d(dim//4, dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_fft = torch.fft.fft2(x, norm='backward')

        center_h, center_w = H//3, W//3
        low_freq = torch.fft.fftshift(x_fft, dim=(-2, -1))
        low_freq_center = low_freq[..., center_h:2*center_h, center_w:2*center_w]
        low_freq_resized = F.interpolate(low_freq_center.real, size=(H, W), 
                                       mode='bicubic', align_corners=False)
        low_feat = self.low_freq_branch(low_freq_resized)

        high_freq = x_fft.clone()

        mask = torch.ones_like(x_fft)
        mask[..., center_h:2*center_h, center_w:2*center_w] = 0.3
        high_freq = high_freq * mask
        high_freq_real = torch.fft.ifft2(high_freq, norm='backward').real
        high_feat = self.high_freq_branch(high_freq_real)
        
 
        fused = self.enhancement(torch.cat([low_feat, high_feat], dim=1))
        
     
        gate_weight = self.gate(fused)
        return fused * gate_weight
        


class MSFE(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        
        self.long_range = EfficientLargeKernel(dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

      
        self.freq_attention = FDN(dim)
       
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(dim*4, dim//2, 1),
            nn.GELU(),
            nn.Conv2d(dim//2, 4, 1),
            nn.Softmax(dim=1)
        )
        
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.in_conv(x)
      
        identity = x
        long_range_feat = self.long_range(out)
        local_feat = self.dw_11(out)
        freq_feat = self.freq_attention(out)
   
        all_feats = torch.cat([
            identity.unsqueeze(1),
            long_range_feat.unsqueeze(1), 
            local_feat.unsqueeze(1),
            freq_feat.unsqueeze(1)
        ], dim=1)  # [B, 4, C, H, W]
        
        fusion_weights = self.fusion_gate(
            all_feats.flatten(1, 2)  # [B, 4*C, H, W]
        ).unsqueeze(2)  # [B, 4, 1, H, W]
        
        fused = (all_feats * fusion_weights).sum(dim=1)
        
        out = self.act(fused)
        return self.out_conv(out)


class DFF(nn.Module):
    def __init__(self, ch_dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias,
                 lin_ch=512):
        super(DFF, self).__init__()

        self.ch_dim = ch_dim
        self.num_heads = num_heads
        self.LayerNorm_type = LayerNorm_type
        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias
        self.lin_ch = lin_ch
      
        self.text_proj_channel = nn.Sequential(
            nn.Linear(self.lin_ch, self.lin_ch // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.lin_ch // 2, self.ch_dim)
        )
        
        self.text_proj_spatial = nn.Sequential(
            nn.Linear(self.lin_ch, self.lin_ch // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.lin_ch // 2, 16 * 3)  
        )

        self.text_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.ch_dim, max(1, self.ch_dim // 4), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, self.ch_dim // 4), self.ch_dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.conv_C = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.ch_dim, 16 * self.ch_dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_H = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(self.ch_dim, 16, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_W = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(self.ch_dim, 16, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * self.ch_dim, self.ch_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.ch_dim),
            nn.ReLU(inplace=True)
        )

        self.dw_conv_original = nn.Conv2d(self.ch_dim, self.ch_dim, kernel_size=3, 
                                         padding=1, groups=self.ch_dim, bias=self.bias)
        self.dw_conv_shuffled = nn.Conv2d(self.ch_dim, self.ch_dim, kernel_size=3,
                                         padding=1, groups=self.ch_dim, bias=self.bias)
        self.concat_conv = nn.Conv2d(2 * self.ch_dim, self.ch_dim, kernel_size=1, 
                                    padding=0, bias=self.bias)
        
        self.msfe = MSFE(self.ch_dim)
    
        self.norm1 = LayerNorm(self.ch_dim, self.LayerNorm_type)
        self.norm2 = LayerNorm(self.ch_dim, self.LayerNorm_type)
        self.norm3 = LayerNorm(self.ch_dim, self.LayerNorm_type)
        self.ffn = FeedForward(self.ch_dim, self.ffn_expansion_factor, self.bias)

    def forward(self, img_featur, text_code):

        img_feature_original = img_featur
        b, c, h, w = img_featur.shape

        text_weights_channel = self.text_proj_channel(text_code).view(b, self.ch_dim, 1, 1)
        text_attention = self.text_channel_attention(img_featur) * text_weights_channel
        branch1_output = img_featur * text_attention

        spatial_params = self.text_proj_spatial(text_code)  # [b, 16*3]
        spatial_scale_c, spatial_scale_h, spatial_scale_w = spatial_params.chunk(3, dim=1)
        
        # 通道维度注意力（文本引导）
        s0_3_c = self.conv_C(img_featur).view(b, 16, -1, 1, 1)
        s0_3_c = s0_3_c * spatial_scale_c.view(b, 16, 1, 1, 1)
        s0_3_c = s0_3_c.mean(1)
        
        # 高度维度注意力（文本引导）
        s0_3_h = self.conv_H(img_featur).view(b, 16, 1, -1, 1)
        s0_3_h = s0_3_h * spatial_scale_h.view(b, 16, 1, 1, 1)
        s0_3_h = s0_3_h.mean(1)
        
        # 宽度维度注意力（文本引导）
        s0_3_w = self.conv_W(img_featur).view(b, 16, 1, 1, -1)
        s0_3_w = s0_3_w * spatial_scale_w.view(b, 16, 1, 1, 1)
        s0_3_w = s0_3_w.mean(1)
        
  
        cube_attention = s0_3_c * s0_3_h * s0_3_w
        branch2_output = img_featur * cube_attention

        fused_features = torch.cat([branch1_output, branch2_output], dim=1)
        restored_feature = self.fusion_conv(fused_features)

        dw_original = self.dw_conv_original(img_feature_original)
        dw_restored = self.dw_conv_shuffled(restored_feature)

        concat_features = torch.cat([dw_original, dw_restored], dim=1)

        adjusted_features = self.concat_conv(concat_features)

        msfe_output = self.msfe(adjusted_features)

        att = msfe_output + img_feature_original  
        output = att + self.ffn(self.norm3(att))

        return output, img_feature_original
        
        

class Topm_CrossAttention_Restormer(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Topm_CrossAttention_Restormer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x_q, x_kv):
        b,c,h,w = x_q.shape
        q = self.q_dwconv(self.q(x_q))
        kv = self.kv_dwconv(self.kv(x_kv))
        k,v = kv.chunk(2, dim=1)  
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        index = torch.topk(attn, k=int(C*9/10), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))
        attn4 = attn4.softmax(dim=-1)
        out4 = (attn4 @ v)
        out =  out4 * self.attn4  
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
  
class DFFIR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [2,3,3,4], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        device = "cuda:1",
        # decoder = False,
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):
        super(ChannelShuffle_skip_textguaid, self).__init__()    
        self.device = device        

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_shuffle_channel1 =DFF(ch_dim = dim,num_heads=heads[0],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias) # encoder level1 shuffle

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_shuffle_channel2 = DFF(ch_dim = int(dim*2**1),num_heads=heads[1],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias) # encoder level2 shuffle

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_shuffle_channel3 = DFF(ch_dim = int(dim*2**2),num_heads=heads[2],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias) # encoder level3 shuffle  

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent_shuffle_channel = DFF(ch_dim = int(dim*2**3),num_heads=heads[3],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias) # latent latent shuffle

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])    

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        #### For Dual-Pixel Defocus Deblurring Task ####
        # self.dual_pixel_task = dual_pixel_task
        # if self.dual_pixel_task:
        #     self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        # ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img ,text_code): # ,text_code
        # text_code = torch.randn(1,512).to(self.device) # 这个在测试模型参数量和计算量时候加上    
        inp_enc_level1 = self.patch_embed(inp_img) # ch 3-->dim:48
        out_enc_level1 = self.encoder_level1(inp_enc_level1) # ch dim:48-->dim:48
        
        inp_enc_level2 = self.down1_2(out_enc_level1) # ch dim:48-->dim*2:96
        out_enc_level2 = self.encoder_level2(inp_enc_level2) # ch dim*2:96-->dim*2:96

        inp_enc_level3 = self.down2_3(out_enc_level2) # ch dim*2:96-->dim*2*2:192
        out_enc_level3 = self.encoder_level3(inp_enc_level3) # ch dim*2*2:192-->dim*2*2:192

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
        latent,_ = self.latent_shuffle_channel(latent,text_code) # latent latent shuffle
                           
        inp_dec_level3 = self.up4_3(latent)
        outt1,_ = self.encoder_shuffle_channel3(out_enc_level3,text_code)
        inp_dec_level3 = torch.cat([inp_dec_level3, outt1], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        outt2,_ = self.encoder_shuffle_channel2(out_enc_level2,text_code)
        inp_dec_level2 = torch.cat([inp_dec_level2, outt2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        outt3,_ = self.encoder_shuffle_channel1(out_enc_level1,text_code)
        inp_dec_level1 = torch.cat([inp_dec_level1, outt3], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)      
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        # if self.dual_pixel_task:
        #     out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        #     out_dec_level1 = self.output(out_dec_level1)
        # ###########################
        # else:
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1



##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [2,3,3,4], 
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1   



