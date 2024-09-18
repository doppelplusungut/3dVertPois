"""
Adapted from MonAI https://docs.monai.io/en/stable/_modules/monai/networks/nets/densenet.html
Key changes: The final flattening and out layers are removed, as the model is used to generate a (downsized) feature map.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class _DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module(
            "norm1",
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels),
        )
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module(
            "conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False)
        )

        self.layers.add_module(
            "norm2",
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels),
        )
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module(
            "conv2",
            conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(
                spatial_dims,
                in_channels,
                growth_rate,
                bn_size,
                dropout_prob,
                act=act,
                norm=norm,
            )
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module(
            "norm",
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels),
        )
        self.add_module("relu", get_act_layer(name=act))
        self.add_module(
            "conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class HeatmapDenseNet(nn.Module):
    """
    Densenet based on: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.
    This network is non-deterministic When `spatial_dims` is 3 and CUDA is enabled. Please check the link below
    for more details:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        n_landmarks: int,
        init_features: int = 64,
        feature_l: int = 256,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_landmarks = n_landmarks
        self.feature_l = feature_l

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[
            Conv.CONV, spatial_dims
        ]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[
            Pool.MAX, spatial_dims
        ]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        conv_type(
                            in_channels,
                            init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    (
                        "norm0",
                        get_norm_layer(
                            name=norm, spatial_dims=spatial_dims, channels=init_features
                        ),
                    ),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5",
                    get_norm_layer(
                        name=norm, spatial_dims=spatial_dims, channels=in_channels
                    ),
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims,
                    in_channels=in_channels,
                    out_channels=_out_channels,
                    act=act,
                    norm=norm,
                )
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # Final convolution to produce n_landmarks heatmaps + feature_l feature maps simultaneously
        self.features.add_module(
            "conv_final",
            conv_type(in_channels, n_landmarks + feature_l, kernel_size=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()  # Ensure input is float
        x = self.features(
            x
        )  # (batch_size, n_landmarks + feature_l, *spatial_shape // (len(block_config)**2))
        # Split the output into landmark heatmaps and feature maps
        heatmaps, feature_map = x.split([self.n_landmarks, self.feature_l], dim=1)

        # Normalize the heatmaps so that they sum to 1 by applying spatial softmax
        B, N, *spatial_shape = heatmaps.shape
        normalized_heatmaps = heatmaps.view(B, N, -1)
        normalized_heatmaps = torch.softmax(normalized_heatmaps, dim=-1)
        normalized_heatmaps = normalized_heatmaps.view(B, N, *spatial_shape)

        # Apply the heatmaps to the feature maps to get the feature encoding for each landmark
        # Expand feature map dimensions from (B, F, H, W, D) to (B, 1, F, H, W, D) for broadcasting
        feature_map_expanded = feature_map.unsqueeze(1)

        # Perform weighted sum
        # heatmaps_normalized is (B, N, H, W, D)
        # feature_map_expanded is (B, 1, F, H, W, D)
        # We want the output to be (B, N, F), summing over the spatial dimensions (H, W, D)
        weighted_features = (
            normalized_heatmaps.unsqueeze(2).detach() * feature_map_expanded
        ).sum(dim=(3, 4, 5))

        return normalized_heatmaps, weighted_features
