import math

import torch
import torch.nn as nn

from utils import (
    ConformerCNNNormalization,
    ModelArchitecture,
    TransformerPositionEncoding,
)


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
        return fused_features


class FeedForwardModule(nn.Module):
    """Macaron-style feed-forward module used by the Conformer block."""

    def __init__(
        self,
        channels,
        expansion_factor,
        dropout,
        residual_scale=0.5,
    ):
        super().__init__()

        hidden_channels = channels * expansion_factor
        self.residual_scale = float(residual_scale)
        self.layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        return tokens + self.residual_scale * self.layers(tokens)


class RelativePositionBias2D(nn.Module):
    """Learn one attention bias for each two-dimensional relative offset."""

    def __init__(self, height, width, num_heads):
        super().__init__()

        self.height = int(height)
        self.width = int(width)
        relative_position_count = (
            (2 * self.height - 1) * (2 * self.width - 1)
        )
        self.bias_table = nn.Parameter(
            torch.zeros(relative_position_count, num_heads)
        )

        rows = torch.arange(self.height)
        columns = torch.arange(self.width)
        coordinates = torch.stack(
            torch.meshgrid(rows, columns, indexing="ij")
        )
        flattened_coordinates = coordinates.flatten(start_dim=1)
        relative_coordinates = (
            flattened_coordinates[:, :, None]
            - flattened_coordinates[:, None, :]
        )
        relative_coordinates[0] += self.height - 1
        relative_coordinates[1] += self.width - 1
        relative_coordinates[0] *= 2 * self.width - 1
        relative_position_index = relative_coordinates.sum(dim=0)
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self):
        token_count = self.height * self.width
        bias = self.bias_table[
            self.relative_position_index.reshape(-1)
        ]
        bias = bias.reshape(token_count, token_count, -1)
        return bias.permute(2, 0, 1).unsqueeze(0)


class ConformerRelativePositionEncoding1D(nn.Module):
    """Transformer-XL-style relative sinusoidal attention for a token sequence."""

    def __init__(self, token_count, channels, num_heads):
        super().__init__()

        if channels % 2 != 0:
            raise ValueError("Conformer positional channels must be even")

        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.scale = self.head_channels ** -0.5
        self.position_projection = nn.Linear(
            channels, channels, bias=False
        )
        self.content_bias = nn.Parameter(
            torch.zeros(num_heads, self.head_channels)
        )
        self.position_bias = nn.Parameter(
            torch.zeros(num_heads, self.head_channels)
        )

        relative_positions = torch.arange(
            -(token_count - 1), token_count, dtype=torch.float32
        )
        inverse_frequencies = torch.exp(
            torch.arange(0, channels, 2, dtype=torch.float32)
            * (-math.log(10000.0) / channels)
        )
        sinusoidal_embeddings = torch.zeros(
            2 * token_count - 1, channels
        )
        angles = relative_positions[:, None] * inverse_frequencies[None, :]
        sinusoidal_embeddings[:, 0::2] = torch.sin(angles)
        sinusoidal_embeddings[:, 1::2] = torch.cos(angles)

        token_positions = torch.arange(token_count)
        relative_position_index = (
            token_positions[:, None]
            - token_positions[None, :]
            + token_count
            - 1
        )
        self.register_buffer(
            "sinusoidal_embeddings",
            sinusoidal_embeddings,
            persistent=False,
        )
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

    def forward(self, queries, keys):
        projected_positions = self.position_projection(
            self.sinusoidal_embeddings
        )
        projected_positions = projected_positions.reshape(
            projected_positions.shape[0],
            self.num_heads,
            self.head_channels,
        )
        pairwise_positions = projected_positions[
            self.relative_position_index
        ]

        content_scores = torch.einsum(
            "bhid,bhjd->bhij",
            queries + self.content_bias[None, :, None, :],
            keys,
        )
        position_scores = torch.einsum(
            "bhid,ijhd->bhij",
            queries + self.position_bias[None, :, None, :],
            pairwise_positions,
        )
        return (content_scores + position_scores) * self.scale


class RelativeMultiHeadSelfAttention(nn.Module):
    """Pre-normalized MHSA with configurable relative position encoding."""

    def __init__(
        self,
        channels,
        num_heads,
        height,
        width,
        dropout,
        position_encoding=TransformerPositionEncoding.LEARNED_2D,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"{channels} channels are not divisible by {num_heads} heads"
            )

        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.scale = self.head_channels ** -0.5
        self.position_encoding = TransformerPositionEncoding(
            position_encoding
        )
        self.normalization = nn.LayerNorm(channels)
        self.qkv_projection = nn.Linear(channels, 3 * channels)
        if self.position_encoding == TransformerPositionEncoding.LEARNED_2D:
            self.relative_position_bias = RelativePositionBias2D(
                height, width, num_heads
            )
            self.conformer_relative_position = None
        elif self.position_encoding == TransformerPositionEncoding.CONFORMER_1D:
            self.relative_position_bias = None
            self.conformer_relative_position = (
                ConformerRelativePositionEncoding1D(
                    height * width, channels, num_heads
                )
            )
        else:
            raise ValueError(
                "Unsupported Transformer position encoding: "
                f"{self.position_encoding}"
            )
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(channels, channels)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        residual = tokens
        normalized_tokens = self.normalization(tokens)
        batch_size, token_count, channels = normalized_tokens.shape

        qkv = self.qkv_projection(normalized_tokens)
        qkv = qkv.reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_channels,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)

        if self.position_encoding == TransformerPositionEncoding.LEARNED_2D:
            attention_scores = queries @ keys.transpose(-2, -1)
            attention_scores = attention_scores * self.scale
            attention_scores = (
                attention_scores + self.relative_position_bias()
            )
        else:
            attention_scores = self.conformer_relative_position(
                queries, keys
            )
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended_tokens = attention_weights @ values
        attended_tokens = attended_tokens.transpose(1, 2).reshape(
            batch_size, token_count, channels
        )
        attended_tokens = self.output_projection(attended_tokens)
        attended_tokens = self.output_dropout(attended_tokens)
        return residual + attended_tokens


class ChannelLayerNorm2d(nn.Module):
    """Apply LayerNorm across channels at every spatial position."""

    def __init__(self, channels):
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, features):
        features = features.permute(0, 2, 3, 1)
        features = self.normalization(features)
        return features.permute(0, 3, 1, 2)


class ConformerCNNModule(nn.Module):
    """Two-dimensional local feature module inside the Transformer branch."""

    def __init__(
        self,
        channels,
        height,
        width,
        dropout,
        normalization=ConformerCNNNormalization.BATCH_NORM,
    ):
        super().__init__()

        self.channels = channels
        self.height = int(height)
        self.width = int(width)
        self.normalization_type = ConformerCNNNormalization(normalization)
        self.normalization = nn.LayerNorm(channels)
        self.input_projection = nn.Conv2d(
            channels, 2 * channels, kernel_size=1
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_convolution = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        if self.normalization_type == ConformerCNNNormalization.BATCH_NORM:
            self.batch_normalization = nn.BatchNorm2d(channels)
        else:
            self.batch_normalization = ChannelLayerNorm2d(channels)
        self.activation = nn.SiLU()
        self.output_projection = nn.Conv2d(
            channels, channels, kernel_size=1
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        residual = tokens
        batch_size, token_count, channels = tokens.shape
        expected_token_count = self.height * self.width
        if token_count != expected_token_count or channels != self.channels:
            raise ValueError(
                "Unexpected token shape for the Conformer CNN module: "
                f"{tuple(tokens.shape)}"
            )

        feature_map = self.normalization(tokens)
        feature_map = feature_map.transpose(1, 2).reshape(
            batch_size, channels, self.height, self.width
        )
        feature_map = self.input_projection(feature_map)
        feature_map = self.glu(feature_map)
        feature_map = self.depthwise_convolution(feature_map)
        feature_map = self.batch_normalization(feature_map)
        feature_map = self.activation(feature_map)
        feature_map = self.output_projection(feature_map)
        feature_map = self.dropout(feature_map)

        output_tokens = feature_map.flatten(start_dim=2).transpose(1, 2)
        return residual + output_tokens


class TransformerBranch(nn.Module):
    """Conformer-style non-local branch that preserves BCHW shape."""

    def __init__(
        self,
        channels,
        spatial_size,
        *,
        num_heads,
        expansion_factor=4,
        dropout=0.1,
        ffn_residual_scale=0.5,
        position_encoding=TransformerPositionEncoding.LEARNED_2D,
        cnn_normalization=ConformerCNNNormalization.BATCH_NORM,
    ):
        super().__init__()

        self.channels = channels
        self.height = int(spatial_size)
        self.width = int(spatial_size)
        self.feed_forward_1 = FeedForwardModule(
            channels,
            expansion_factor,
            dropout,
            residual_scale=ffn_residual_scale,
        )
        self.self_attention = RelativeMultiHeadSelfAttention(
            channels,
            num_heads,
            self.height,
            self.width,
            dropout,
            position_encoding=position_encoding,
        )
        self.cnn_module = ConformerCNNModule(
            channels,
            self.height,
            self.width,
            dropout,
            normalization=cnn_normalization,
        )
        self.feed_forward_2 = FeedForwardModule(
            channels,
            expansion_factor,
            dropout,
            residual_scale=ffn_residual_scale,
        )
        self.final_normalization = nn.LayerNorm(channels)

    def forward(self, features):
        batch_size, channels, height, width = features.shape
        expected_shape = (self.channels, self.height, self.width)
        if (channels, height, width) != expected_shape:
            raise ValueError(
                "Unexpected Transformer input shape: "
                f"{tuple(features.shape)}"
            )

        tokens = features.flatten(start_dim=2).transpose(1, 2)
        tokens = self.feed_forward_1(tokens)
        tokens = self.self_attention(tokens)
        tokens = self.cnn_module(tokens)
        tokens = self.feed_forward_2(tokens)
        tokens = self.final_normalization(tokens)
        return tokens.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )


class CTBlock(nn.Module):
    """Parallel CNN-Transformer block described by the paper."""

    def __init__(
        self,
        channels,
        spatial_size,
        *,
        transformer_heads,
        transformer_expansion_factor=4,
        transformer_dropout=0.1,
        transformer_ffn_residual_scale=0.5,
        transformer_position_encoding=TransformerPositionEncoding.LEARNED_2D,
        transformer_cnn_normalization=ConformerCNNNormalization.BATCH_NORM,
    ):
        super().__init__()

        self.cnn_branch = MultiscaleCNNBlock(channels)
        self.transformer_branch = TransformerBranch(
            channels,
            spatial_size,
            num_heads=transformer_heads,
            expansion_factor=transformer_expansion_factor,
            dropout=transformer_dropout,
            ffn_residual_scale=transformer_ffn_residual_scale,
            position_encoding=transformer_position_encoding,
            cnn_normalization=transformer_cnn_normalization,
        )
        self.fusion = nn.Conv2d(
            2 * channels, channels, kernel_size=1
        )

    def forward(self, features):
        cnn_features = self.cnn_branch(features)
        transformer_features = self.transformer_branch(features)
        concatenated_features = torch.cat(
            [cnn_features, transformer_features], dim=1
        )
        fused_features = self.fusion(concatenated_features)
        return features + fused_features


class ChannelAttention(nn.Module):
    """Lightweight channel attention with a residual connection."""

    def __init__(self, kernel_size=3):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("Channel attention kernel size must be odd")

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_convolution = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, features):
        pooled_features = self.global_pool(features)
        channel_descriptor = pooled_features.squeeze(-1).transpose(1, 2)
        channel_weights = self.channel_convolution(channel_descriptor)
        channel_weights = torch.sigmoid(channel_weights)
        channel_weights = channel_weights.transpose(1, 2).unsqueeze(-1)
        return features + features * channel_weights


class SpatialAttention(nn.Module):
    """Spatial attention reconstructed from Section 2.4 of the paper."""

    def __init__(self):
        super().__init__()

        self.extrema_branch = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=1),
        )
        self.distribution_branch = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=1),
        )
        self.fusion = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, features):
        maximum = features.amax(dim=1, keepdim=True)
        minimum = features.amin(dim=1, keepdim=True)
        mean = features.mean(dim=1, keepdim=True)
        standard_deviation = features.std(
            dim=1, keepdim=True, correction=0
        )

        extrema_features = self.extrema_branch(
            torch.cat([maximum, minimum], dim=1)
        )
        distribution_features = self.distribution_branch(
            torch.cat([mean, standard_deviation], dim=1)
        )
        spatial_weights = torch.sigmoid(
            self.fusion(
                torch.cat(
                    [extrema_features, distribution_features], dim=1
                )
            )
        )
        return features + features * spatial_weights


class ChannelSpatialAttentionBlock(nn.Module):
    """Sequential channel-spatial attention that preserves BCHW shape."""

    def __init__(self, channel_kernel_size=3):
        super().__init__()

        self.channel_attention = ChannelAttention(channel_kernel_size)
        self.spatial_attention = SpatialAttention()

    def forward(self, features):
        channel_refined_features = self.channel_attention(features)
        return self.spatial_attention(channel_refined_features)


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
        cnn_features = features + self.cnn_block(features)
        pooled_features = self.global_pool(cnn_features)
        pooled_features = pooled_features.flatten(start_dim=1)
        return self.classifier(pooled_features)


class CNNCSAClassifier(nn.Module):
    """CNN classifier followed by channel-spatial attention."""

    def __init__(self, input_channels, feature_channels, num_classes):
        super().__init__()

        self.input_projection = nn.Conv2d(
            input_channels, feature_channels, kernel_size=1
        )
        self.cnn_block = MultiscaleCNNBlock(feature_channels)
        self.csa_block = ChannelSpatialAttentionBlock()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_channels, num_classes)

    def forward(self, patches):
        features = self.input_projection(patches)
        cnn_features = features + self.cnn_block(features)
        refined_features = self.csa_block(cnn_features)
        pooled_features = self.global_pool(refined_features)
        pooled_features = pooled_features.flatten(start_dim=1)
        return self.classifier(pooled_features)


class CNNTransformerClassifier(nn.Module):
    """Classifier using the paper's parallel CNN-Transformer block."""

    def __init__(
        self,
        input_channels,
        feature_channels,
        num_classes,
        patch_size,
        *,
        transformer_heads,
        transformer_expansion_factor=4,
        transformer_dropout=0.1,
        transformer_ffn_residual_scale=0.5,
        transformer_position_encoding=TransformerPositionEncoding.LEARNED_2D,
        transformer_cnn_normalization=ConformerCNNNormalization.BATCH_NORM,
    ):
        super().__init__()

        self.input_projection = nn.Conv2d(
            input_channels, feature_channels, kernel_size=1
        )
        self.ct_block = CTBlock(
            feature_channels,
            patch_size,
            transformer_heads=transformer_heads,
            transformer_expansion_factor=transformer_expansion_factor,
            transformer_dropout=transformer_dropout,
            transformer_ffn_residual_scale=transformer_ffn_residual_scale,
            transformer_position_encoding=transformer_position_encoding,
            transformer_cnn_normalization=transformer_cnn_normalization,
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_channels, num_classes)

    def forward(self, patches):
        features = self.input_projection(patches)
        ct_features = self.ct_block(features)
        pooled_features = self.global_pool(ct_features)
        pooled_features = pooled_features.flatten(start_dim=1)
        return self.classifier(pooled_features)


class CNNTransformerCSAClassifier(nn.Module):
    """CNN-Transformer classifier followed by channel-spatial attention."""

    def __init__(
        self,
        input_channels,
        feature_channels,
        num_classes,
        patch_size,
        *,
        transformer_heads,
        transformer_expansion_factor=4,
        transformer_dropout=0.1,
        transformer_ffn_residual_scale=0.5,
        transformer_position_encoding=TransformerPositionEncoding.LEARNED_2D,
        transformer_cnn_normalization=ConformerCNNNormalization.BATCH_NORM,
    ):
        super().__init__()

        self.input_projection = nn.Conv2d(
            input_channels, feature_channels, kernel_size=1
        )
        self.ct_block = CTBlock(
            feature_channels,
            patch_size,
            transformer_heads=transformer_heads,
            transformer_expansion_factor=transformer_expansion_factor,
            transformer_dropout=transformer_dropout,
            transformer_ffn_residual_scale=transformer_ffn_residual_scale,
            transformer_position_encoding=transformer_position_encoding,
            transformer_cnn_normalization=transformer_cnn_normalization,
        )
        self.csa_block = ChannelSpatialAttentionBlock()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_channels, num_classes)

    def forward(self, patches):
        features = self.input_projection(patches)
        ct_features = self.ct_block(features)
        refined_features = self.csa_block(ct_features)
        pooled_features = self.global_pool(refined_features)
        pooled_features = pooled_features.flatten(start_dim=1)
        return self.classifier(pooled_features)


class CNNTransformerCSAAdaptivePruningClassifier(
    CNNTransformerCSAClassifier
):
    """CTA-Net with a Transformer-conditioned, trainable spectral gate.

    A first pass predicts one soft gate per input band from the Transformer
    branch. The gate is applied to the original patch, which is then classified
    in a second pass. The soft sigmoid keeps ordinary gradients flowing during
    training; a dataset-level binary mask can be exported after training.
    """

    def __init__(
        self,
        input_channels,
        feature_channels,
        num_classes,
        patch_size,
        *,
        transformer_heads,
        transformer_expansion_factor=4,
        transformer_dropout=0.1,
        transformer_ffn_residual_scale=0.5,
        transformer_position_encoding=TransformerPositionEncoding.LEARNED_2D,
        transformer_cnn_normalization=ConformerCNNNormalization.BATCH_NORM,
        pruning_gate_slope=20.0,
        pruning_sparsity_weight=0.001,
    ):
        super().__init__(
            input_channels=input_channels,
            feature_channels=feature_channels,
            num_classes=num_classes,
            patch_size=patch_size,
            transformer_heads=transformer_heads,
            transformer_expansion_factor=transformer_expansion_factor,
            transformer_dropout=transformer_dropout,
            transformer_ffn_residual_scale=transformer_ffn_residual_scale,
            transformer_position_encoding=transformer_position_encoding,
            transformer_cnn_normalization=transformer_cnn_normalization,
        )
        self.input_channels = int(input_channels)
        self.feature_channels = int(feature_channels)
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)
        self.transformer_heads = int(transformer_heads)
        self.transformer_expansion_factor = int(
            transformer_expansion_factor
        )
        self.transformer_dropout = float(transformer_dropout)
        self.transformer_ffn_residual_scale = float(
            transformer_ffn_residual_scale
        )
        self.transformer_position_encoding = TransformerPositionEncoding(
            transformer_position_encoding
        )
        self.transformer_cnn_normalization = ConformerCNNNormalization(
            transformer_cnn_normalization
        )
        self.pruning_gate_slope = float(pruning_gate_slope)
        self.pruning_sparsity_weight = float(pruning_sparsity_weight)
        if self.pruning_gate_slope <= 0:
            raise ValueError("Pruning gate slope must be positive")
        if self.pruning_sparsity_weight < 0:
            raise ValueError("Pruning sparsity weight cannot be negative")

        self.band_gate_head = nn.Linear(feature_channels, input_channels)
        nn.init.zeros_(self.band_gate_head.weight)
        nn.init.constant_(self.band_gate_head.bias, 0.15)

    def predict_band_gates(self, patches):
        """Predict differentiable per-sample gates from Transformer features."""
        features = self.input_projection(patches)
        transformer_features = self.ct_block.transformer_branch(features)
        pooled = transformer_features.mean(dim=(2, 3))
        gate_logits = self.band_gate_head(pooled)
        return torch.sigmoid(self.pruning_gate_slope * gate_logits)

    def forward(self, patches):
        gates = self.predict_band_gates(patches)
        self._last_pruning_penalty = (
            self.pruning_sparsity_weight
            * gates.sum(dim=1).mean()
        )
        masked_patches = patches * gates[:, :, None, None]
        return super().forward(masked_patches)

    def pruning_regularization(self, patches):
        """Return the configured sparsity penalty for the current batch."""
        del patches
        return getattr(self, "_last_pruning_penalty", 0.0)

    @torch.no_grad()
    def estimate_global_band_mask(self, loader, device, threshold=0.5):
        """Aggregate gates over a supplied loader into one fixed band mask."""
        self.eval()
        gate_sum = torch.zeros(self.input_channels, device=device)
        sample_count = 0
        for patches, _ in loader:
            patches = patches.to(device)
            gates = self.predict_band_gates(patches)
            gate_sum += gates.sum(dim=0)
            sample_count += gates.shape[0]
        if sample_count == 0:
            raise ValueError("Cannot estimate a band mask from an empty loader")
        mean_gates = gate_sum / sample_count
        selected_indices = torch.nonzero(
            mean_gates >= threshold, as_tuple=True
        )[0]
        if selected_indices.numel() == 0:
            selected_indices = mean_gates.argmax().reshape(1)
        return selected_indices.cpu(), mean_gates.cpu()


class FrozenBandPrunedClassifier(nn.Module):
    """Compact CTA-Net that selects a fixed set of original input bands."""

    def __init__(self, classifier, selected_band_indices):
        super().__init__()
        self.classifier = classifier
        self.register_buffer(
            "selected_band_indices",
            torch.as_tensor(selected_band_indices, dtype=torch.long),
        )

    def forward(self, patches):
        selected_patches = patches.index_select(
            1, self.selected_band_indices
        )
        return self.classifier(selected_patches)


def build_compact_pruned_model(adaptive_model, selected_band_indices):
    """Export a standard CTA-Net with its first projection narrowed to K bands."""
    selected_band_indices = torch.as_tensor(
        selected_band_indices,
        dtype=torch.long,
        device=adaptive_model.input_projection.weight.device,
    ).flatten()
    if selected_band_indices.numel() == 0:
        raise ValueError("At least one spectral band must be retained")
    if (
        selected_band_indices.min() < 0
        or selected_band_indices.max() >= adaptive_model.input_channels
    ):
        raise ValueError("Selected band index is outside the input spectrum")

    compact = CNNTransformerCSAClassifier(
        input_channels=selected_band_indices.numel(),
        feature_channels=adaptive_model.feature_channels,
        num_classes=adaptive_model.num_classes,
        patch_size=adaptive_model.patch_size,
        transformer_heads=adaptive_model.transformer_heads,
        transformer_expansion_factor=(
            adaptive_model.transformer_expansion_factor
        ),
        transformer_dropout=adaptive_model.transformer_dropout,
        transformer_ffn_residual_scale=(
            adaptive_model.transformer_ffn_residual_scale
        ),
        transformer_position_encoding=(
            adaptive_model.transformer_position_encoding
        ),
        transformer_cnn_normalization=(
            adaptive_model.transformer_cnn_normalization
        ),
    )
    compact.load_state_dict(
        {
            name: value
            for name, value in adaptive_model.state_dict().items()
            if not name.startswith("input_projection.")
            and not name.startswith("band_gate_head.")
        },
        strict=False,
    )
    with torch.no_grad():
        compact.input_projection.weight.copy_(
            adaptive_model.input_projection.weight.index_select(
                1, selected_band_indices
            )
        )
        compact.input_projection.bias.copy_(
            adaptive_model.input_projection.bias
        )
    return FrozenBandPrunedClassifier(
        compact, selected_band_indices.cpu()
    )


def build_model(
    architecture,
    *,
    input_channels,
    feature_channels,
    num_classes,
    patch_size,
    transformer_heads,
    transformer_expansion_factor=4,
    transformer_dropout=0.1,
    transformer_ffn_residual_scale=0.5,
    transformer_position_encoding=TransformerPositionEncoding.LEARNED_2D,
    transformer_cnn_normalization=ConformerCNNNormalization.BATCH_NORM,
    pruning_gate_slope=20.0,
    pruning_sparsity_weight=0.001,
):
    """Build the model selected by the experiment configuration."""
    if architecture == ModelArchitecture.CNN_ONLY:
        return CNNOnlyBaseline(
            input_channels=input_channels,
            feature_channels=feature_channels,
            num_classes=num_classes,
        )
    if architecture == ModelArchitecture.CNN_CSA:
        return CNNCSAClassifier(
            input_channels=input_channels,
            feature_channels=feature_channels,
            num_classes=num_classes,
        )
    if architecture == ModelArchitecture.CNN_TRANSFORMER:
        return CNNTransformerClassifier(
            input_channels=input_channels,
            feature_channels=feature_channels,
            num_classes=num_classes,
            patch_size=patch_size,
            transformer_heads=transformer_heads,
            transformer_expansion_factor=transformer_expansion_factor,
            transformer_dropout=transformer_dropout,
            transformer_ffn_residual_scale=transformer_ffn_residual_scale,
            transformer_position_encoding=transformer_position_encoding,
            transformer_cnn_normalization=transformer_cnn_normalization,
        )
    if architecture == ModelArchitecture.CNN_TRANSFORMER_CSA:
        return CNNTransformerCSAClassifier(
            input_channels=input_channels,
            feature_channels=feature_channels,
            num_classes=num_classes,
            patch_size=patch_size,
            transformer_heads=transformer_heads,
            transformer_expansion_factor=transformer_expansion_factor,
            transformer_dropout=transformer_dropout,
            transformer_ffn_residual_scale=transformer_ffn_residual_scale,
            transformer_position_encoding=transformer_position_encoding,
            transformer_cnn_normalization=transformer_cnn_normalization,
        )
    if architecture == ModelArchitecture.CNN_TRANSFORMER_CSA_ADAPTIVE_PRUNING:
        return CNNTransformerCSAAdaptivePruningClassifier(
            input_channels=input_channels,
            feature_channels=feature_channels,
            num_classes=num_classes,
            patch_size=patch_size,
            transformer_heads=transformer_heads,
            transformer_expansion_factor=transformer_expansion_factor,
            transformer_dropout=transformer_dropout,
            transformer_ffn_residual_scale=(
                transformer_ffn_residual_scale
            ),
            transformer_position_encoding=transformer_position_encoding,
            transformer_cnn_normalization=(
                transformer_cnn_normalization
            ),
            pruning_gate_slope=pruning_gate_slope,
            pruning_sparsity_weight=pruning_sparsity_weight,
        )
    raise ValueError(f"Unsupported model architecture: {architecture}")
