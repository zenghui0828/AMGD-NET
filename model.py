import torch
import torch.nn as nn
import torch.nn.functional as F

# 激活函数使用 SiLU
ACT = nn.SiLU
# ---------------------------
# CBAM Attention Module
# ---------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x

# ---------------------------
# CBAM + Dilated Conv Block
# ---------------------------
class CBAM_DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, reduction=16):
        super(CBAM_DilatedConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.dilated_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels, reduction=reduction)

    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.cbam(x)
        return x

# Squeeze-and-Excitation 模块
class SE(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ECA(nn.Module):
    def __init__(self, channels,k_size=3):
        super(ECA,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y =y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

# Fused MBConv 块（没有分离卷积，将扩展卷积和深度卷积分离掉）
class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion,use_se=True):
        super(FusedMBConv, self).__init__()
        self.stride = stride
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion

        # 扩展卷积
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size,
                                     stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = ACT()
        self.attn = ECA(hidden_dim) if use_se else nn.Identity()
        # 投影卷积
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.act(self.bn1(self.expand_conv(x)))
        out = self.bn2(self.project_conv(out))
        if self.use_res_connect:
            return x + out
        else:
            return out

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion,use_se=True):
        super(FusedMBConv1, self).__init__()
        self.stride = stride
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion

        # 大小卷积核
        self.large_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size,
                                     stride=stride, padding=kernel_size//2, bias=False)
        self.small_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1,
                                    stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = ACT()

        self.attn = ECA(hidden_dim) if use_se else nn.Identity()
        # 投影卷积
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.large_conv(x) + self.small_conv(x)
        out = self.act(self.bn1(out))
        out = self.bn2(self.project_conv(out))
        if self.use_res_connect:
            return x + out
        else:
            return out
# 标准的 MBConv 块（包含深度可分离卷积和 SE 模块）
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion, use_se=True):
        super(MBConv, self).__init__()
        self.stride = stride
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion

        # 第一步：扩展（1x1 卷积）
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = ACT()

        # 第二步：深度卷积（3x3 或 5x5 卷积，根据 kernel_size）
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                 stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # 可选的 SE 模块
        self.attn = ECA(hidden_dim) if use_se else nn.Identity()

        # 第三步：投影（1x1 卷积）
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.act(self.bn1(self.expand_conv(x)))
        out = self.act(self.bn2(self.dw_conv(out)))
        out = self.attn(out)
        out = self.bn3(self.project_conv(out))
        if self.use_res_connect:
            return x + out
        else:
            return out


class AFF_ChannelWise(nn.Module):
    def __init__(self, height, width):
        super(AFF_ChannelWise, self).__init__()
        self.height = height
        self.width = width

        self.fc1 = nn.Sequential(
            nn.Linear(2, 1),
            nn.LayerNorm(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        assert C == 3, "AFF_ChannelWise 只支持三通道输入"

        out = []
        for c in range(C):
            xi = x[:, c:c + 1, :, :]  # [B,1,H,W]
            avg_pool = nn.AdaptiveAvgPool2d(1)(xi).view(B, 1)
            max_pool = nn.AdaptiveMaxPool2d(1)(xi).view(B, 1)
            stats = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2]

            attn = self.fc1(stats).view(B, 1, 1, 1)  # [B,1,1,1]
            xi_weighted = xi * attn
            out.append(xi_weighted)

        out = torch.cat(out, dim=1)  # [B,3,H,W]
        return out

class GatedHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(GatedHead, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.SiLU()
        )
        self.output = nn.Linear(in_features, num_classes)

    def forward(self, x):
        gated = self.gate(x)*self.transform(x)
        return self.output(gated)

# EfficientNetV2 主网络
class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(EfficientNetV2, self).__init__()
        #self.aff = AFF_ChannelWise(height=224,width=224)

        # 下面的配置主要参考 EfficientNetV2-S 版本
        # 每个配置项格式：[expansion, out_channels, num_blocks, stride, kernel_size, block_type]
        self.cfgs = [
            # Fused MBConv 阶段
            [1, 24, 2, 1, 3, 'fused'],
            [4, 48, 4, 2, 3, 'fused'],
            [4, 64, 4, 2, 3, 'conv1'],
            # MBConv 阶段
            [4, 128, 6, 2, 3, 'conv1'],
            [6, 160, 9, 1, 3, 'mbconv'],
            [6, 256, 15, 2, 3, 'mbconv'],
        ]
        # 第一个卷积层（stem）
        out_channels = int(24 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            ACT()
        )

        # 构建各个阶段的卷积块
        layers = []
        in_channels = out_channels
        for expansion, c, n, s, k, block_type in self.cfgs:
            out_channels = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                if block_type == 'fused':
                    layers.append(FusedMBConv(in_channels, out_channels, kernel_size=k, stride=stride, expansion=expansion))
                elif block_type == 'conv1':
                    layers.append(Conv1(in_channels, out_channels, kernel_size=k, stride=stride, expansion=expansion))
                else:
                    layers.append(MBConv(in_channels, out_channels, kernel_size=k, stride=stride, expansion=expansion))
                in_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.cbam_block = CBAM_DilatedConvBlock(in_channels, in_channels)

        # Head 部分
        head_channels = int(1280 * width_mult)
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            ACT(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.head = GatedHead(in_features=head_channels, num_classes=num_classes)
        #self.head = nn.Sequential(
        #    nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False),
        #    nn.BatchNorm2d(head_channels),
        #    ACT(),
        #    nn.AdaptiveAvgPool2d(1),
        #    nn.Flatten(),
        #    nn.Linear(head_channels, num_classes)
        #)
        self._initialize_weights()


    def _initialize_weights(self):
        """权重初始化函数：对卷积层使用 kaiming_normal 初始化，
        对 BN 层设置权重为 1，偏置为 0，对全连接层使用正态分布初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.aff(x)
        x = self.stem(x)
        x = self.features(x)
        x = self.cbam_block(x)
        x = self.conv_head(x)
        x = self.head(x)
        return x