# TODO imports

import math

params_one = {
    "padding": 1,
    "kernel_size": 3,
    "stride": 1,
    "dilation": 1,
    "padding_mode": "relpicate",
    "bias": True,
}

params_two = {
    "padding": 3,
    "kernel_size": 7,
    "stride": 1,
    "dilation": 1,
    "padding_mode": "relpicate",
    "bias": True,
}


def pow_of_two(x):
    n = math.log(x, 2)
    if x == 2 ** int(n):
        return True, int(n)
    return False, n


class EncoderV(nn.Module):
    def __init__(self, dims, n):
        super(EncoderV, self).__init__()
        self.dims = dims
        self.fc = nn.Sequential(nn.Linear(dims * 2, dims * 2, bias=True), nn.ReLU())

        layers = [nn.Conv2d(3, dims // n, **params_two)]
        for i in range(n - 1):
            layers += [
                nn.Conv2d(dims // (n - i), dims // (n - i - 1), **params_one),
                nn.ReLU(),
                nn.AvgPool2d((2, 2)),
            ]

        layers += [
            nn.Conv2d(dims // (n - i - 1), dims * 2, **params_one),
            nn.AvgPool2d((2, 2)),
            nn.ReLU(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        B = x.shape[0]
        x = self.layers(x)  # BxCxHxW -> BxCx1x1
        x = x.view(B, self.dims * 2)  # BxDims
        x = self.fc(x)
        mu, log_sigma = x.chunk(2, dim=1)
        return mu, log_sigma


class DecoderV(nn.Module):
    """
    (∗,C×r^2,H,W) to a tensor of shape (∗,C,H×r,W×r)
    """

    def __init__(self, dims, HxW):
        super(DecoderV, self).__init__()
        self.dims = dims
        # (Hin+2×padding−dilation×(kernel_size−1)−1)/stride + 1

        is_power, n_output = pow_of_two(HxW)
        assert is_power, "Output size is not power of two"
        is_power, n_dims = pow_of_two(dims)
        assert is_power, "Dim size is not power of two"
        assert n_output <= n_dims, "Output size power is greater than dim size power"

        layers = []
        for i in range(n_output):
            layers += [
                nn.Conv2d(dims, dims * 2, **params_one),
                nn.ReLU(),
                nn.PixelShuffle(2),
            ]
            dims = dims // 2

        layers += [nn.Conv2d(dims, dims, **params_one), nn.ReLU()]
        layers.append(nn.Conv2d(dims, 3, **params_one))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        B = z.shape[0]
        z = z.view(B, self.dims, 1, 1)  # BxDimsx1x1
        z = self.layers(z)  # BxCx1x1 - > Bx3xHxW
        return z


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


class VAE(nn.Module):
    def __init__(self, HxW, dims):
        super(VAE, self).__init__()

        _, n = pow_of_two(HxW)
        self.encoder = EncoderV(dims, int(n))
        self.decoder = DecoderV(dims, HxW)
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0], requires_grad=True)

        # self.encoder = ConvEncoder((3,32,32),dims)
        # self.decoder =  ConvDecoder((3,32,32),dims)

    def gaussian_nll(self, mu, log_sigma, x):
        return (
            0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
            + log_sigma
            + 0.5 * np.log(2 * np.pi)
        )

    def forward(self, x):
        mu, log_var = self.encoder(x)
        # gaussian_sampler
        z = torch.randn_like(mu) * torch.exp(0.5 * log_var) + mu
        x_z = self.decoder(z)
        log_sigma = softclip(self.log_sigma, -6)
        recon_loss = self.gaussian_nll(x, log_sigma, x_z).sum()

        kl_loss = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum()
        # kl_loss = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) #.sum(1).mean()
        # print(kl_loss.shape())

        #         kl_loss = -log_sigma - 0.5 + (torch.exp(2 * log_sigma) + mu.pow(2)) * 0.5
        #         kl_loss = kl_loss.sum(1).mean()

        return {
            "rec_img": x_z,
            "kl": kl_loss,
            "recon_loss": recon_loss,
            "total_loss": kl_loss + recon_loss,
        }
