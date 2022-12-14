import torch
from torch import nn


class AnalysisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=2):
        super().__init__()
        dilation = (1, dilation_rate, dilation_rate)

        self.layers = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels, 3, dilation=dilation, padding="same"),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, 3, dilation=dilation, padding="same"),
            nn.ELU(),
            nn.GroupNorm(8, out_channels),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super().__init__()
        dilation = (1, dilation_rate, dilation_rate)
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.elu = nn.ELU()
        
        self.layers = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels, 3, dilation=dilation, padding="same"),
            nn.ELU(),
            nn.Conv3d(out_channels, out_channels, 3, dilation=dilation, padding="same"),
            nn.ELU(),
            nn.GroupNorm(8, out_channels),
        ])
        
    def forward(self, inputs):
        x, skip_x = inputs
        x = self.elu(self.upconv(x))
        x = torch.cat([x, skip_x], 1)  # channel concat

        for layer in self.layers:
            x = layer(x)
        return x


class OutputLayer(nn.Module):
    """
    The final outputs of the model. Can vary based on the task.
    Multiple outputs with different activation functions are supported.
    """
    def __init__(self, activations):
        """
        activations is a dict of activation functions to be applied.
        Specify None if no activation is to be applied
        """
        super().__init__()
        self.activations = activations
        self.fc_layers = nn.ModuleDict({key: nn.Conv3d(8, 1, 1, padding="same")
                                        for key in activations})

    def forward(self, x):
        pre_act = {key: layer(x) for key, layer in self.fc_layers.items()}

        return {
            key: pre_act[key] if act is None else act(pre_act[key])
            for key, act in self.activations.items()
        }


class ClipLayer(nn.Module):
    def __init__(self, min_val=-1.0, max_val=1.0):
        self.min = min_val
        self.max = max_val

    def forward(self, x):
        return torch.clip(x, self.min, self.max)
