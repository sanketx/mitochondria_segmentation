from torch import nn
from .base_module import BaseModule
from .blocks import AnalysisBlock, SynthesisBlock


class Basic3DUNet(BaseModule):
    def __init__(self, output_layer, **kwargs):
        super(Basic3DUNet, self).__init__(**kwargs)
        self.output_layer = output_layer

    def build(self):
        self.pool = nn.MaxPool3d(2)
        self.bottom_layer = AnalysisBlock(128, 256)
        
        self.analysis_layers = nn.ModuleList([
            AnalysisBlock(1, 8),
            AnalysisBlock(8, 16),
            AnalysisBlock(16, 32),
            AnalysisBlock(32, 64),
            AnalysisBlock(64, 128),
        ])
        
        self.synthesis_layers = nn.ModuleList([
            SynthesisBlock(256, 128),
            SynthesisBlock(128, 64),
            SynthesisBlock(64, 32),
            SynthesisBlock(32, 16),
            SynthesisBlock(16, 8),
        ])

    def forward(self, x):
        analysis_outputs = []

        for layer in self.analysis_layers:
            x = layer(x)
            analysis_outputs.append(x)
            x = self.pool(x)

        x = self.bottom_layer(x)

        for layer, skip_x in zip(self.synthesis_layers, analysis_outputs[::-1]):
            x = layer([x, skip_x])

        return self.output_layer(x)  # dict of tensors


if __name__ == '__main__':
    pass
