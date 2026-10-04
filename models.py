import torch
import torch.nn as nn


class MultiscaleCNNBlock(nn.Module):
    """CNN branch of the paper's CT block."""

    def __init__(self, channels):
        super().__init__()

        self.branch_1x1 = nn.Conv2d(
            channels, channels, kernel_size=1
        )
        self.branch_3x3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.branch_two_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.branch_three_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        self.fusion = nn.Conv2d(
            4 * channels, channels, kernel_size=1
        )

    def forward(self, features):
        branch_features = [
            self.branch_1x1(features),
            self.branch_3x3(features),
            self.branch_two_3x3(features),
            self.branch_three_3x3(features),
        ]
        concatenated_features = torch.cat(branch_features, dim=1)
        fused_features = self.fusion(concatenated_features)
        return features + fused_features


class CNNOnlyBaseline(nn.Module):
    """CNN-only baseline derived from the CNN branch of the CT block."""

    def __init__(self, input_channels, feature_channels, num_classes):
        super().__init__()

        self.input_projection = nn.Conv2d(
            input_channels, feature_channels, kernel_size=1
        )
        self.cnn_block = MultiscaleCNNBlock(feature_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_channels, num_classes)

    def forward(self, patches):
        features = self.input_projection(patches)
        cnn_features = self.cnn_block(features)

        # Level 3: pass `features` to the Transformer in parallel here,
        # then concatenate its output with `cnn_features`.

        pooled_features = self.global_pool(cnn_features)
        pooled_features = pooled_features.flatten(start_dim=1)
        return self.classifier(pooled_features)
