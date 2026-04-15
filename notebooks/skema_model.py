from abc import ABCMeta, abstractmethod

import segmentation_models_pytorch as smp
import torch

EPS = 1e-10


class SKeMaModelBase(torch.nn.Module, metaclass=ABCMeta):
    in_channels = 10

    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="tu-maxvit_tiny_tf_512",
            in_channels=self.in_channels,
            encoder_weights=None,
        )

        self.register_buffer(
            "per_channel_mean",
            torch.zeros((1, self.in_channels, 1, 1)),
        )

        self.register_buffer(
            "per_channel_std",
            torch.ones((1, self.in_channels, 1, 1)),
        )

    @staticmethod
    def normalized_index(a, b):
        return (a - b) / (a + b + EPS)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")


class SKeMaModel(SKeMaModelBase):
    def forward(self, x):
        # Unpack spectral bands
        blue = x.select(1, 0).unsqueeze(1)
        green = x.select(1, 1).unsqueeze(1)
        red = x.select(1, 2).unsqueeze(1)
        nir = x.select(1, 3).unsqueeze(1)
        re = x.select(1, 4).unsqueeze(1)

        # Compute vegetation indices
        ndvi = self.normalized_index(nir, red)
        gndvi = self.normalized_index(nir, green)
        ndvi_re = self.normalized_index(re, red)

        # Compute other indices
        ndwi = self.normalized_index(green, nir)
        chl_green = (nir / (green + EPS)) - 1  # Chlorophyll Index Green

        # Stack all bands and indices
        x_aug = torch.cat([blue, green, red, nir, re, ndvi, ndwi, gndvi, chl_green, ndvi_re], dim=1)

        x_aug_normalized = (x_aug - self.per_channel_mean) / self.per_channel_std

        return self.model(x_aug_normalized)


class SKeMaBathyModel(SKeMaModelBase):
    in_channels = 13

    def forward(self, x):
        # Unpack spectral bands
        blue = x.select(1, 0).unsqueeze(1)
        green = x.select(1, 1).unsqueeze(1)
        red = x.select(1, 2).unsqueeze(1)
        nir = x.select(1, 3).unsqueeze(1)
        re = x.select(1, 4).unsqueeze(1)
        substrate = x.select(1, 5).unsqueeze(1)
        bathymetry = x.select(1, 6).unsqueeze(1)
        slope = x.select(1, 7).unsqueeze(1)

        # Compute vegetation indices
        ndvi = self.normalized_index(nir, red)
        gndvi = self.normalized_index(nir, green)
        ndvi_re = self.normalized_index(re, red)

        # Compute other indices
        ndwi = self.normalized_index(green, nir)
        chl_green = (nir / (green + EPS)) - 1  # Chlorophyll Index Green

        # Stack all bands and indices
        x_aug = torch.cat(
            [blue, green, red, nir, re, substrate, bathymetry, slope, ndvi, ndwi, gndvi, chl_green, ndvi_re], dim=1
        )

        x_aug_normalized = (x_aug - self.per_channel_mean) / self.per_channel_std

        return self.model(x_aug_normalized)
