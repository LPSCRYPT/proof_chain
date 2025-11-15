#!/usr/bin/env python3
"""
Flexible GAN architectures for comprehensive testing
Supports all 30 configurations with different optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleGenerator(nn.Module):
    """Flexible generator that can be configured for different architectures"""

    def __init__(self, config):
        super(FlexibleGenerator, self).__init__()
        self.config = config
        arch = config['architecture']

        self.latent_dim = arch.get('latent_dim', 100)
        self.embed_dim = arch.get('embed_dim', 50)
        self.ngf = arch.get('ngf', 48)
        self.num_layers = arch.get('num_layers', 4)
        self.use_bias = arch.get('use_bias', False)
        self.activation = arch.get('activation', 'relu')

        # Handle different model types
        self.model_type = arch.get('model_type', 'conv')

        # Embedding layer
        self.label_embedding = nn.Embedding(10, self.embed_dim)

        if self.model_type == 'mlp':
            self._build_mlp_model(arch)
        else:
            self._build_conv_model(arch)

    def _build_mlp_model(self, arch):
        """Build MLP-only model for ultra-minimal ops"""
        hidden_dims = arch.get('hidden_dims', [512, 1024, 2048, 3072])

        layers = []
        in_dim = self.latent_dim + self.embed_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim, bias=self.use_bias),
                self._get_activation(),
                nn.BatchNorm1d(hidden_dim) if arch.get('use_batchnorm', True) else nn.Identity(),
            ])
            in_dim = hidden_dim

        # Final layer to image
        layers.append(nn.Linear(in_dim, 3 * 32 * 32))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def _build_conv_model(self, arch):
        """Build convolutional model"""
        # Adaptive ngf (can be list or single value)
        if isinstance(self.ngf, list):
            ngf_list = self.ngf
        else:
            ngf_list = [self.ngf] * self.num_layers

        # Initial projection
        self.proj = nn.Sequential(
            nn.Linear(self.latent_dim + self.embed_dim, ngf_list[0] * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf_list[0] * 8 * 4 * 4) if arch.get('use_batchnorm', True) else nn.Identity(),
            self._get_activation()
        )

        # Build layers based on num_layers
        layers = []
        in_channels = ngf_list[0] * 8

        for i in range(self.num_layers - 1):
            out_channels = ngf_list[min(i+1, len(ngf_list)-1)] * (4 // (2**i)) if i < 3 else ngf_list[-1]
            out_channels = max(out_channels, ngf_list[-1])  # Don't go below min channels

            # Regular conv transpose
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=self.use_bias))

            # Normalization
            if arch.get('normalization', 'batch') == 'instance':
                layers.append(nn.InstanceNorm2d(out_channels))
            elif arch.get('use_batchnorm', True):
                layers.append(nn.BatchNorm2d(out_channels))

            # Activation
            layers.append(self._get_activation())

            # Optional dropout
            if arch.get('dropout', 0) > 0:
                layers.append(nn.Dropout2d(arch['dropout']))

            in_channels = out_channels

        # Final layer
        layers.append(nn.ConvTranspose2d(in_channels, 3, 4, 2, 1, bias=self.use_bias))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

        # Optional skip connections
        self.use_skip = arch.get('use_skip', False)
        if self.use_skip:
            self.skip_proj = nn.Conv2d(ngf_list[0] * 8, 3, 1, bias=False)

    def _get_activation(self):
        """Get activation function based on config"""
        if self.activation == 'relu':
            return nn.ReLU(inplace=True)
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif self.activation == 'gelu':
            return nn.GELU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU(inplace=True)

    def forward(self, noise, labels):
        # Embed labels
        label_embed = self.label_embedding(labels)

        if self.model_type == 'mlp':
            # Flatten and concatenate
            x = torch.cat([noise.view(noise.size(0), -1), label_embed], dim=1)
            x = self.main(x)
            return x.view(x.size(0), 3, 32, 32)
        else:
            # Concatenate noise and label embedding
            x = torch.cat([noise.view(noise.size(0), -1), label_embed], dim=1)
            x = self.proj(x)
            x = x.view(x.size(0), -1, 4, 4)

            if self.use_skip:
                skip = self.skip_proj(x)
                skip = F.interpolate(skip, size=(32, 32), mode='bilinear', align_corners=False)

            x = self.main(x)

            if self.use_skip:
                x = x + skip
                x = torch.tanh(x)

            return x


class FlexibleDiscriminator(nn.Module):
    """Flexible discriminator matching generator architecture"""

    def __init__(self, config):
        super(FlexibleDiscriminator, self).__init__()
        self.config = config
        arch = config['architecture']

        self.ngf = arch.get('ngf', 48)
        self.embed_dim = arch.get('embed_dim', 50)
        self.use_spectral_norm = arch.get('use_spectral_norm', False)

        # Build discriminator
        ndf = self.ngf  # Use same base filters as generator

        layers = []

        # Input layer
        if self.use_spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1, bias=False)))
        else:
            layers.append(nn.Conv2d(3, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Hidden layers
        in_channels = ndf
        for i in range(3):
            out_channels = min(ndf * (2 ** (i+1)), 512)

            if self.use_spectral_norm:
                layers.append(nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)))
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))

            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        self.main = nn.Sequential(*layers)

        # Output layers
        self.conv_final = nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False)

        # Conditional projection
        self.label_embedding = nn.Embedding(10, self.embed_dim)
        self.label_proj = nn.Linear(self.embed_dim, in_channels, bias=False)

    def forward(self, x, labels):
        features = self.main(x)

        # Class conditioning
        label_embed = self.label_embedding(labels)
        label_proj = self.label_proj(label_embed)
        label_proj = label_proj.view(label_proj.size(0), -1, 1, 1)

        # Combine with features
        features = features * label_proj

        # Final output
        output = self.conv_final(features)
        return output.view(-1, 1)


def create_generator(config):
    """Factory function to create generator based on config"""
    return FlexibleGenerator(config)


def create_discriminator(config):
    """Factory function to create discriminator based on config"""
    return FlexibleDiscriminator(config)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_einsum_ops(model):
    """Estimate einsum operations for ZK circuit"""
    ops = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            ops += 1
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            ops += 2  # Approximate
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            ops += 1
        elif isinstance(module, nn.Embedding):
            ops += 1

    # Adjust based on architecture specifics
    if hasattr(model, 'config'):
        arch = model.config.get('architecture', {})
        if arch.get('use_attention'):
            ops += 5
        if arch.get('use_skip'):
            ops += 2
        if arch.get('multi_scale'):
            ops += 3

    return ops