import torch
from torch import nn


class Basic3DUNet(nn.Module):
    def __init__(self, **kwargs):
        super(Basic3DUNet, self).__init__(**kwargs)
        self.output_layer = nn.Conv3d(16, 1, 1, padding="same")

        self.bottom_layer = nn.Sequential(
            nn.Conv3d(128, 128, (2, 3, 3), padding="same"),
            nn.Conv3d(128, 128, (2, 3, 3), padding="same"),
        )

        self.analysis_layers = nn.ModuleList([
            AnalysisBlock(1, 16, stride=4),
            AnalysisBlock(16, 64, stride=4),
            AnalysisBlock(64, 128, stride=4),
        ])

        self.synthesis_layers = nn.ModuleList([
            SynthesisBlock(128, 128, 128, stride=4),
            SynthesisBlock(128, 64, 64, stride=4),
            SynthesisBlock(64, 16, 16, stride=4),
        ])

    def forward(self, x):
        analysis_outputs = []

        for layer in self.analysis_layers:
            x, y = layer(x)
            analysis_outputs.append(y)

        x = self.bottom_layer(x)

        for layer, skip_x in zip(self.synthesis_layers, analysis_outputs[::-1]):
            x = layer([x, skip_x])

        x = self.output_layer(x)
        return torch.sigmoid(x)


class AnalysisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.pool = nn.Conv3d(out_channels, out_channels, stride, stride=stride)

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding="same"),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding="same"),
            nn.GELU(),
            nn.GroupNorm(8, out_channels),
        )

    def forward(self, x):
        x = self.layers(x)
        y = self.pool(x)
        return y, x


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, stride):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, stride, stride=stride)
        self.gelu = nn.GELU()

        self.layers = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, 3, dilation=1, padding="same"),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, dilation=2, padding="same"),
            nn.GELU(),
            nn.GroupNorm(8, out_channels),
        )

    def forward(self, inputs):
        x, skip_x = inputs
        x = self.gelu(self.upconv(x))
        x = torch.cat([x, skip_x], 1)
        x = self.layers(x)
        return x
