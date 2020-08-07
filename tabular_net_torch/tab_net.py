"""
Inspired by
https://github.com/awslabs/autogluon/blob/eacee5bacf0e03b2131922f77f32413b05c2b9e2/autogluon/utils/tabular/ml/models/tabular_nn/embednet.py
"""

import torch
import torch.nn as nn


class CatEmbed(nn.Module):
    def __init__(self, params):
        self.embed = nn.ModuleDict()
        for fe_name in range(params["cat_features"]):
            # different embedding size for each feature
            self.embed[fe_name] = nn.Embedding(
                params["cat_embed"]["fe_name"]["num_categories"],
                params["cat_embed"]["fe_name"]["embedding_dim"],
            )

    def forward(self, x_cat):
        feature_vector = []
        for fe_name in x_cat:
            # EMBED SHAPE: BATCH x 1 x EMBED_DIM (Always one element for a column)
            feature_vector.append(self.embed[fe_name](x_cat[fe_name]))
            return torch.cat(
                feature_vector, dim=2
            ).squeeze()  # SHAPE: BATCH x SUM(EMBED DIMS)


class DenseResidual(nn.Module):
    def __init__(self, params):
        # Bias is rebundant if use with batch norm or layer norm, we shift values anyway
        # Or set elementwise_affine=False in norm layers
        bias = max(params["batch_norm"], params["layer_norm"])
        self.main_block = [
            nn.Linear(params["mid_features"], params["mid_features"], bias=bias)
        ]
        if params["batch_norm"] == params["layer_norm"]:
            print("Can't use Layer & Batch norm simultanisouly")
            return None
        if params["batch_norm"]:
            self.main_block.append(nn.BatchNorm1d(params["mid_features"]))
        if params["layer_norm"]:
            self.main_block.append(nn.LayerNorm(params["mid_features"]))
        if params["drop_out"] > 0:
            self.main_block.append(nn.Dropout(params["drop_out"]))
        self.main_block.append(nn.ReLU())
        self.main_block = nn.Sequential(self.main_block)

    def forward(self, x):
        return x + self.main_block(x)


class TabNet(nn.Module):
    def __init__(self, params):
        self.params = params
        self.linear_collector = nn.Linear(
            params["input_concat_vector_size"], params["mid_features"]
        )
        if params["num_real"] > 0:
            self.liner_real_values = nn.Linear(
                params["num_real"], params["mid_features"]
            )

        self.categorical_embed = CatEmbed(params)
        self.main = nn.Sequential(
            [DenseResidual] * params["num_residual"]
            + [nn.Linear(params["mid_features"], 1)]
        )

    def forward(self, x):
        x_real = x["real_vector"]
        x_cat = x["cat_vector"]
        cat_vector = self.categorical_embed(x_cat)
        real_vector = self.liner_real_values(x_real)
        vector = torch.cat((cat_vector, real_vector), dim=1)
        out = self.linear_collector(vector)
        out = self.main_block(out)
        return out
