import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  # Ensure mamba-ssm is installed
from utils_.PE_utils import ProgressiveExpansionQKV

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Basic Utility Blocks
class BN_ACT(nn.Module):
    def __init__(self, num_features, act=True):
        super(BN_ACT, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.act = act
        self.relu = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=2):
        super(Conv_Block, self).__init__()
        self.bn_act = BN_ACT(in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation
        )

    def forward(self, x):
        x = self.bn_act(x)
        x = self.conv(x)
        return x

class SE_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=2, reduction=16):
        super(SE_Conv_Block, self).__init__()
        self.conv_block = Conv_Block(in_channels, out_channels, stride, dilation)
        self.se = SEBlock(out_channels, reduction=reduction)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.se(x)
        return x

# Mamba-based Global Context Module
class MambaGlobalContext(nn.Module):
    def __init__(self, channels, d_state=8, d_conv=4):
        super(MambaGlobalContext, self).__init__()
        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=1  # Reduced from 2 for memory efficiency
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]
        x_mamba = self.mamba(x_flat)  # [B, H*W, C]
        x_mamba = self.norm(x_mamba)
        x_mamba = x_mamba.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]
        return x + x_mamba

# Enhanced Attention Fusion
class EnhancedAttentionFusion(nn.Module):
    def __init__(self, in_global, in_local, out_channels):
        super().__init__()
        self.global_transform = nn.Conv2d(in_global, out_channels, kernel_size=1)
        self.local_transform = nn.Conv2d(in_local, out_channels, kernel_size=1)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_channels * 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.channel_attn = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, global_feat, local_feat):
        g = self.global_transform(global_feat)
        l = self.local_transform(local_feat)
        x = torch.cat([g, l], dim=1)
        spatial_alpha = self.spatial_attn(x)
        channel_alpha = self.channel_attn(x)
        fused = g * spatial_alpha * channel_alpha + l * (1 - spatial_alpha) * (1 - channel_alpha)
        fused = self.fusion_conv(torch.cat([g, fused], dim=1))
        return fused

# Multihead Attention Block with Progressive Expansion (Simplified)
class Conv_Block_PE_Att(nn.Module):
    def __init__(self, in_channels, out_channels, heads, num_dim, stride=1, expand_type=1, function='arctan'):
        super().__init__()
        self.bn_act = BN_ACT(in_channels)
        self.conv = nn.Conv2d(in_channels, num_dim, kernel_size=1, stride=stride)
        self.pe_block = ProgressiveExpansionQKV(
            in_channels=num_dim,
            out_channels=num_dim,
            expand_type=expand_type,
            function=function
        )
        self.attention = nn.Conv2d(num_dim, num_dim, kernel_size=1)  # Simplified attention
        self.post_conv = nn.Conv2d(num_dim, out_channels, kernel_size=1) if out_channels != num_dim else nn.Identity()

    def forward(self, x):
        x = self.bn_act(x)
        x_conv = self.conv(x)
        q, k, v = self.pe_block(x_conv)
        attn_output = self.attention(v)  # Simplified attention with 1x1 conv
        out = x_conv + attn_output
        out = self.post_conv(out)
        return out

# Residual Block with PE Attention
class Residual_Block_PE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=2, dropout_p=0.0, se_reduction=16):
        super(Residual_Block_PE, self).__init__()
        self.conv_block1 = Conv_Block_PE_Att(in_channels, out_channels, heads=2, num_dim=out_channels, stride=stride)
        self.conv_block2 = SE_Conv_Block(out_channels, out_channels, stride=1, dilation=dilation, reduction=se_reduction)
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.shortcut_bn_act = BN_ACT(out_channels)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x):
        res = self.conv_block1(x)
        res = self.conv_block2(res)
        res = self.dropout(res)
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_bn_act(shortcut)
        return res + shortcut

# Stem
class Stem(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(Stem, self).__init__()
        self.conv_initial = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_block = Conv_Block(out_channels, out_channels, stride=1, dilation=dilation)
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.shortcut_bn_act = BN_ACT(out_channels)

    def forward(self, x):
        conv = self.conv_initial(x)
        conv = self.conv_block(conv)
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_bn_act(shortcut)
        return conv + shortcut

# Encoder Block with Mamba
class Encoder_Block_Mamba(nn.Module):
    def __init__(self, in_channels, out_channels, heads, num_dim, stride=1, dilation=2, expand_type=1, function='arctan'):
        super(Encoder_Block_Mamba, self).__init__()
        self.conv_block_pe_att = Conv_Block_PE_Att(
            in_channels, out_channels, heads, num_dim, stride, expand_type, function
        )
        self.se_conv_block = SE_Conv_Block(out_channels, out_channels, stride=1, dilation=dilation)
        self.global_context = MambaGlobalContext(out_channels)

    def forward(self, x):
        res = self.conv_block_pe_att(x)
        res = self.se_conv_block(res)
        res = self.global_context(res)
        return res

# UPEN_mamba Model
class UPEN_mamba(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512], num_dim=64, dropout_p=0.2, dilation=2, se_reduction=16, expand_type=1, function='arctan'):
        super(UPEN_mamba, self).__init__()
        f = features
        self.heads = 2  # Hardcoded as per model design

        # Encoders
        self.stem = Stem(in_channels, f[0], dilation=1)
        self.encoder1 = Encoder_Block_Mamba(f[0], f[1], self.heads, num_dim, stride=2, dilation=dilation, expand_type=expand_type, function=function)
        self.encoder2 = Encoder_Block_Mamba(f[1], f[2], self.heads, num_dim, stride=2, dilation=dilation, expand_type=expand_type, function=function)
        self.encoder3 = Encoder_Block_Mamba(f[2], f[3], self.heads, num_dim, stride=2, dilation=dilation, expand_type=expand_type, function=function)

        # Bridge
        self.bridge_conv1 = SE_Conv_Block(f[3], f[3], stride=1, dilation=dilation, reduction=se_reduction)
        self.dropout1 = nn.Dropout2d(dropout_p)
        self.bridge_mamba = MambaGlobalContext(f[3], d_state=8, d_conv=4)
        self.dropout2 = nn.Dropout2d(dropout_p)

        # Decoders
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att_fusion1 = EnhancedAttentionFusion(f[3], f[2], f[3])
        self.decoder1 = Residual_Block_PE(f[3], f[3], stride=1, dilation=dilation, dropout_p=dropout_p, se_reduction=se_reduction)
        self.aux_conv1 = nn.Conv2d(f[3], 1, kernel_size=1)  # Quarter resolution output

        self.att_fusion2 = EnhancedAttentionFusion(f[3], f[1], f[2])
        self.decoder2 = Residual_Block_PE(f[2], f[2], stride=1, dilation=dilation, dropout_p=dropout_p, se_reduction=se_reduction)
        self.aux_conv2 = nn.Conv2d(f[2], 1, kernel_size=1)  # Half resolution output

        self.att_fusion3 = EnhancedAttentionFusion(f[2], f[0], f[1])
        self.decoder3 = Residual_Block_PE(f[1], f[1], stride=1, dilation=dilation, dropout_p=dropout_p, se_reduction=se_reduction)
        self.final_conv = nn.Conv2d(f[1], 1, kernel_size=1)  # Full resolution output

    def forward(self, x):
        e1 = self.stem(x)
        e2 = self.encoder1(e1)
        e3 = self.encoder2(e2)
        e4 = self.encoder3(e3)
        b = self.bridge_conv1(e4)
        b = self.dropout1(b)
        b = self.bridge_mamba(b)
        b = self.dropout2(b)
        u1 = self.up(b)
        u1 = self.att_fusion1(u1, e3)
        d1 = self.decoder1(u1)
        u2 = self.up(d1)
        u2 = self.att_fusion2(u2, e2)
        d2 = self.decoder2(u2)
        u3 = self.up(d2)
        u3 = self.att_fusion3(u3, e1)
        d3 = self.decoder3(u3)
        out3 = self.final_conv(d3)
        out1 = self.aux_conv1(d1)  # Quarter resolution
        out2 = self.aux_conv2(d2)  # Half resolution
        return out3, out2, out1  # Multi-scale for evaluation
