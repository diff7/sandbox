import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        The mean and standard-deviation are calculated separately
        over the last certain number dimensions which has to be
        set at initialization.
        """

    def __init__(self, C_IN):
        super(CLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(c_in)

    def forward(self, x):
        # x (B, C, W, H)
        x = x.transpose(1, 3).contiguous()  # (B, W, H, C)
        x = self.layer_norm(x)
        return x.transpose(1, 3).contiguous()  # (B, C, W, H)


class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):

        """
        Masks example

        A =  [1., 1., 1.]
             [1., 0., 0.]
             [0., 0., 0.]

        B =  [1., 1., 1.]
             [1., 1., 0.]
             [0., 0., 0.]
        """

        assert mask_type in ("A", "B")
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.zeros_like(self.weight))
        _, _, h_center, w_center = self.mask.size()
        self.mask[:, :, h_center // 2, w_center // 2 + (mask_type == "B") :] = 1
        self.mask[:, :, h_center // 2 + 1 :] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class PixelCNNResBlock(nn.Module):

    """
    (Hin​+2×padding−dilation×(kernel_size−1)−1​)/stride + 1

    Using params below will yield following out sizes:
    H_out = H_in
    W_out = W_in
    """

    def __init__(self, channels):
        super().__init__()

        params = {
            "padding": 3,
            "kernel_size": 7,
            "stride": 1,
            "dilation": 1,
            "padding_mode": "replicate",
            "groups": 1,
        }

        self.block = nn.Sequential(
            LayerNorm(dim),
            nn.ReLU(),
            MaskConv2d("B", channels, channels, **params),
            LayerNorm(channels),
            nn.ReLU(),
            MaskConv2d("B", channels, channels, **params),
            LayerNorm(channels),
            nn.ReLU(),
            MaskConv2d("B", channels, channels, **params),
        )

    def forward(self, x_BxCxHxW):
        return x_BxCxHxW + self.block(x_BxCxHxW)


class PixelCNN(nn.Module):
    def __init__(self, channels, n_layers=5):
        super().__init__()
        layers = [MaskConv2d("A", channels, channels), LayerNorm(dim), nn.ReLU()]

        mid = n_layers // 2
        chan = channels
        for i in range(mid + 1 + mid):
            if i + 1 <= mid:
                k = 1 / i
            if i + 1 >= mid:
                k = i
            chan = int(k * chan)

            #         for i in range(2*mid-1):
            # ...:     if i+1 >=mid: k = -1
            # ...:     if i+1 == mid: k = 0
            # ...:     if i+1<=mid: k = 1
            # ...:     chan = chan*(2**k)
            # ...:     print(chan, k, i)
            #
            layers.append(PixelCNNResBlock(channels))
            layers.append(LayerNorm(channels))
            layers.append(nn.ReLU())

        model = nn.ModuleList(
            [
                MaskConv2d(
                    "A", dim, dim, 7, padding=3, conditional_size=conditional_size
                ),
                LayerNorm(dim),
                nn.ReLU(),
            ]
        )
        for _ in range(n_layers - 1):
            model.append(PixelCNNResBlock(dim, conditional_size=conditional_size))
        model.extend(
            [
                LayerNorm(dim),
                nn.ReLU(),
                MaskConv2d("B", dim, 512, 1, conditional_size=conditional_size),
                nn.ReLU(),
                MaskConv2d("B", 512, code_size, 1, conditional_size=conditional_size),
            ]
        )
        self.net = model
        self.input_shape = input_shape
        self.code_size = code_size

    def forward(self, x, cond=None):

        return out

    def loss(self, x, cond=None):
        return OrderedDict(loss=F.cross_entropy(self(x, cond=cond), x))

    def sample(self, n, cond=None):
        samples = torch.zeros(n, *self.input_shape).long().cuda()
        with torch.no_grad():
            for r in range(self.input_shape[0]):
                for c in range(self.input_shape[1]):
                    logits = self(samples, cond=cond)[:, :, r, c]
                    logits = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(logits, 1).squeeze(-1)
        return samples
