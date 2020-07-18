import torch
import torch.nn as nn

# TODO: Change predictor to every N block
# TODO: Concat predictors outputs and add FC on top of that


class ResidualBlock(nn.Module):
    def __init__(self, c_in, groups=1):
        super().__init__()

        """
        (Hin​+2×padding−dilation×(kernel_size−1)−1​)/stride + 1

        Using params below will yield following out sizes:
        H_out = H_in
        W_out = W_in
        """

        params = {
            "padding": 1,
            "kernel_size": 3,
            "stride": 1,
            "dilation": 1,
            "padding_mode": "replicate",
            "groups": groups if c_in % groups == 0 else 1,
        }

        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_in, **params),
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_in, **params),
        )

    def forward(self, x_BxCxHxW):
        return x_BxCxHxW + self.net(x_BxCxHxW)


class CLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, C_IN):
        super(CLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(C_IN)

    def forward(self, x):
        # x (B, C, W, H)
        x = x.transpose(1, 3).contiguous()  # (B, W, H, C)
        x = self.layer_norm(x)
        return x.transpose(1, 3).contiguous()  # (B, C, W, H)


class Aorta(nn.Module):
    """
    (Hin​+2×padding−dilation×(kernel_size−1)−1​)/stride + 1

    H_out = H_in //4, W_out = W_in //4
    """

    def __init__(self, c_in, c_mid, c_out, groups=4):
        super().__init__()

        params = {
            "padding": 1,
            "kernel_size": 3,
            "stride": 2,
            "dilation": 1,
            "padding_mode": "replicate",
            "groups": groups if c_in % groups == 0 else 1,
        }

        self.mp = nn.AdaptiveMaxPool2d(1)
        self.convs = nn.Sequential(
            nn.Conv2d(c_in, c_mid, **params),
            CLayerNorm(c_mid),
            nn.SELU(),
            nn.Conv2d(c_mid, c_out, **params),
            nn.SELU(),
        )

    def forward(self, x_BxCxWxH):
        B = x_BxCxWxH.shape[0]
        x_BxCxWxH = self.convs(x_BxCxWxH)
        x_BxCx1x1 = self.mp(x_BxCxWxH)
        return x_BxCxWxH, x_BxCx1x1


class Predictor(nn.Module):
    def __init__(self, in_C, n_classes):
        super().__init__()
        self.in_C = in_C
        self.fc = nn.Sequential(nn.Linear(in_C, n_classes), nn.ReLU())

    def forward(self, x_BxC):
        B = x_BxC.shape[0]
        x_BxC = x_BxC.view(B, self.in_C)
        x_BxN_classes = self.fc(x_BxC)
        return x_BxN_classes


class LoveNet(nn.Module):
    def __init__(
        self,
        c_in,
        n_classes=2,
        num_blocks=4,
        use_mid_predictor=False,
        c_mid=128,
        groups=4,
    ):
        super().__init__()

        self.use_mid_predictor = use_mid_predictor
        self.num_blocks = num_blocks
        self.main_stream = nn.ModuleList()
        self.predictors = nn.ModuleList()

        self.predictors.append(Predictor(c_mid, n_classes))
        for i in range(num_blocks):
            self.main_stream.append(Aorta(c_in, c_mid, c_mid, groups))
            c_in = c_mid
            if use_mid_predictor and i != 0:
                self.predictors.append(Predictor(c_mid, n_classes))

    def forward(self, x_BxCxWxH):
        x_BxCx1x1 = torch.zeros_like(x_BxCxWxH)
        preds_BxN = []
        for i in range(self.num_blocks):
            if (i + 1) % 4 == 0:
                x_BxCxWxH = x_BxCxWxH + prev_BxCx1x1
            if (i + 3) % 4 == 0:
                prev_BxCx1x1 = x_BxCx1x1
            x_BxCxWxH, x_BxCx1x1 = self.main_stream[i](x_BxCxWxH)

            if self.use_mid_predictor and i != self.num_blocks - 1:
                preds_BxN.append(self.predictors[i](x_BxCx1x1))
                if i == 0:
                    preds_BxN_final = preds_BxN[i]
                else:
                    preds_BxN_final += preds_BxN[i]

        # collect predictions  from outputs of each block
        # and average them
        if self.use_mid_predictor:
            preds_BxN_final += self.predictors[i](x_BxCx1x1)
        else:
            preds_BxN_final = self.predictors[0](x_BxCx1x1)

        return preds_BxN_final
