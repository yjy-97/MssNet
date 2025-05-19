import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

class SSC(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        print('[SCC]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.SSC = SSC(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, _, _ = x.shape

        gamma = 1
        x = x + self.drop_path(gamma * self.SSC(self.norm1(x)))

        shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = shortcut + self.drop_path(x)
        return x


class spa(nn.Module):
    def __init__(self):
        super(spa, self).__init__()

        self.conv1 = nn.Conv2d(3, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, c, _, _ = x.shape

        output = torch.ones_like(x)
        for i in range(c):
            x_cut = x[:, i:i + 1, :, :]
            avg_out = torch.mean(x_cut, dim=1, keepdim=True)
            max_out, _ = torch.max(x_cut, dim=1, keepdim=True)
            min_out, _ = torch.min(x_cut, dim=1, keepdim=True)
            x_cut = torch.cat([avg_out, max_out, min_out], dim=1)
            x_cut = self.conv1(x_cut)
            output[:, i:i + 1, :, :] = x_cut[:, :, :, :]
        sig_out = self.sigmoid(output)
        x = sig_out * x
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        self.sa = spa()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.sa(x)
        x = x + s
        x = self.forward_features(x)
        x = self.head(x)
        return x


def Mssnet_t(num_classes: int):
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def Mssnet_s(num_classes: int):
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def Mssnet_b(num_classes: int):
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model

