import torch
from torch import nn
import torch.nn.functional as F


params_down = {
    "padding": 1,
    "kernel_size": 4,
    "stride": 2,
    "dilation": 1,
    "padding_mode": "replicate",
}


params_up = {
    "padding": 1,
    "kernel_size": 3,
    "stride": 1,
    "dilation": 1,
    "padding_mode": "replicate",
    "bias": True,
}


def my_add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = my_add_module


class goDown(nn.Module):
    """
    HxWout = HxWin // 2
    """

    def __init__(self, dims_in, dims_out):
        super(goDown, self).__init__()
        self.down = nn.Sequential()
        self.down.add(nn.Conv2d(dims_in, dims_out, **params_down))
        self.down.add(nn.ReLU())

    def forward(self, x):
        return self.down(x)


class goUp_pixel(nn.Module):
    """
    HxWout = HxWin // 2
    """

    def __init__(self, dims_in, dims_out):
        super(goUp_pixel, self).__init__()
        self.up = nn.Sequential()
        self.up.add(nn.PixelShuffle(2))
        self.up.add(nn.Conv2d(dims_in // 4, dims_out, **params_up))
        self.up.add(nn.ReLU())

    def forward(self, x):
        return self.up(x)


class goUp_nearest(nn.Module):
    """
    HxWout = HxWin // 2
    """

    def __init__(self, dims_in, dims_out):

        super(goUp_nearest, self).__init__()
        self.up = nn.Sequential()
        self.up.add(nn.Upsample(scale_factor=2, mode="nearest"))
        self.up.add(nn.Conv2d(dims_in, dims_out, **params_up))
        self.up.add(nn.ReLU())

    def forward(self, x):
        return self.up(x)


class UnetLike(nn.Module):
    def __init__(self, scale=4, mode="", dims_in=3):
        super(UnetLike, self).__init__()

        self.layers_down = nn.ModuleList()
        self.layers_up = nn.ModuleList()
        self.scale = scale
        # dims_in = 3

        dims_out = 8
        dims_mid = dims_in
        for i in range(scale):
            dims_out *= 2
            self.layers_down.append(goDown(dims_mid, dims_out))
            dims_mid = dims_out

        self.layers_down.append(nn.Conv2d(dims_mid, dims_mid, **params_up))
        self.layers_down.append(nn.ReLU())

        if mode == "nearest":
            goUp = goUp_nearest
        if mode == "pixel":
            goUp = goUp_pixel

        dims_mid = 2 * dims_out  # 256
        # in   256 128 64 32
        # out   64  32 16  3
        for i in range(scale):
            dims_out = dims_mid // 4 if i + 1 != scale else dims_in
            self.layers_up.append(goUp(dims_mid, dims_out))
            dims_mid = dims_mid // 2

    def forward(self, x):
        down_outputs = []
        for i in range(self.scale):
            x = self.layers_down[i](x)
            down_outputs.append(x)

        x_up = self.layers_down[i + 1](x)  # 128
        # c_down = 8*2**4 = 128
        # c_up = c_down*2 = 256
        down_outputs.reverse()
        for i in range(self.scale):
            x_down = down_outputs[i]  # 128, 64, 32, 16
            x_cat = torch.cat((x_up, x_down), dim=1)  # 256, 128, 64, 32
            x_up = self.layers_up[i](x_cat)  # 128, 64, 32, 3

        return x_up
