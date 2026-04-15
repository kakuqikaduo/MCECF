import types
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.io import savemat

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


######################################################################
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


###########  Network Modules ###############
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """
    Transpose convolution layer to upsample the feature maps
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    """
    Residual convolutional block for feature enhancement in decoder
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out




class OverlapPatchEmbed(nn.Module):
    """
    Patch Embedding layer to donwsample spatial resolution before each stage.
    in_chans: number of channels of input features
    embed_dim: number of channels for output features
    patch_size: kernel size of the convolution
    stride: stride value of convolution
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return x, H, W

################## Encoder Modules ####################
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        # self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        # x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class GLCAM(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)
        self.convx1_1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, groups=dim // 2)
        self.qkvl = nn.Conv2d(dim // 2, (dim // 4) * 4, kernel_size=1, padding=0)
        self.q_conv = nn.Conv2d(dim // 4, dim // 4, kernel_size=1, padding=0)
        self.k_conv = nn.Conv2d(dim // 4, dim // 4, kernel_size=1, padding=0)
        self.v_conv = nn.Conv2d(dim // 4, dim // 4, kernel_size=1, padding=0)
        self.l_conv = nn.Conv2d(dim // 4, dim // 4, kernel_size=1, padding=0)
        self.pool_q = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_k = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_x1_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x1_1, x2 = torch.split(x, [C // 2, C // 2], dim=1)
        x11 = self.dwconv(x1_1)
        x12 = self.convx1_1(self.pool_x1_1(x1_1))
        x1 = self.act(x11 * x12)
        x2_qkvl = self.act(self.qkvl(x2)) 
        x2_qkvl = x2_qkvl.reshape(B, self.heads, C // 4, H, W)
        q = torch.sum(x2_qkvl[:, :-3, :, :, :], dim=1) 
        k = x2_qkvl[:, -3, :, :, :]  
        v = x2_qkvl[:, -2, :, :, :]  
        lfeat = x2_qkvl[:, -1, :, :, :]  
        q = self.act(self.q_conv(q)) 
        k = self.act(self.k_conv(k))  
        v = self.act(self.v_conv(v))  
        lfeat = self.act(self.l_conv(lfeat))  
        q = self.pool_q(q)  #  [B, C//4, 1, 1]
        k = self.pool_k(k)  #  [B, C//4, 1, 1]
        q = q.flatten(2)  #  [B, heads, C//4, H*W]
        k = k.flatten(2)  #  [B, heads, C//4, H*W]
        # 计算qk分数
        qk = torch.matmul(q, k.transpose(1, 2))  
        qk = torch.softmax(qk, dim=1).transpose(1, 2)
        v = v.flatten(2)  # [B, heads, C//4, H*W]
        x2_out = torch.matmul(qk, v).reshape(B, C // 4, H, W)
        x = torch.cat([x1, lfeat, x2_out], dim=1)
        return x


class EncoderBlock(nn.Module):
    """
    dim: number of channels of input features
    """

    def __init__(self, dim, drop_path=0.1, mlp_ratio=4, heads=4):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.glcam = GLCAM(dim, heads=heads)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)


        self.parallel_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act1 = nn.GELU()


        self.parallel_conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act2 = nn.GELU()


        self.layer_norm3 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

    def forward(self, x, H, W):
        B, C, H, W = x.shape
        x_res = x
        x = self.layer_norm1(x)
        x = self.glcam(x)
        x = x + x_res

        x_res = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + x_res

        parallel_res1 = x
        parallel_x1 = self.parallel_conv1(x)
        parallel_x1 = self.act1(parallel_x1)
        parallel_x1 = parallel_x1 + parallel_res1

        parallel_res2 = x
        parallel_x2 = self.parallel_conv2(x)
        parallel_x2 = self.act2(parallel_x2)
        parallel_x2 = parallel_x2 + parallel_res2

        x = x + parallel_x1 + parallel_x2

        x = self.layer_norm3(x)
        return x




class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


################## Encoder #########################
# Transormer Ecoder with x4, x8, x16, x32 scales
class Encoder(nn.Module):
    def __init__(self, patch_size=3, in_chans=3, num_classes=2, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], drop_path_rate=0., heads=[4, 4, 4, 4],
                 depths=[3, 3, 4, 3]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        ############ Stage-1 (x1/4 scale)
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # cur = 0

        self.block1 = nn.ModuleList()
        for i in range(depths[0]):
            self.block1.append(EncoderBlock(dim=embed_dims[0], mlp_ratio=mlp_ratios[0]))

        ############# Stage-2 (x1/8 scale)
        # cur += depths[0]

        self.block2 = nn.ModuleList()
        for i in range(depths[1]):
            self.block2.append(EncoderBlock(dim=embed_dims[1], mlp_ratio=mlp_ratios[1]))

        ############# Stage-3 (x1/16 scale)
        # cur += depths[1]

        self.block3 = nn.ModuleList()
        for i in range(depths[2]):
            self.block3.append(EncoderBlock(dim=embed_dims[2], mlp_ratio=mlp_ratios[2]))

        ############# Stage-4 (x1/32 scale)
        # cur += depths[2]

        self.block4 = nn.ModuleList()
        for i in range(depths[3]):
            self.block4.append(EncoderBlock(dim=embed_dims[3], mlp_ratio=mlp_ratios[3]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        return outs

    def forward(self, x):
        features = self.forward_features(x)
        return features


