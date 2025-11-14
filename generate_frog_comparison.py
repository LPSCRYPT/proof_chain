#!/usr/bin/env python3
"""
Generate frog images from both GAN models for comparison
Frog is class 6 in CIFAR-10
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# TinyGenerator - the simple one with 79.5% accuracy
class TinyGenerator(nn.Module):
    def __init__(self, latent_dim=32, num_classes=10):
        super(TinyGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, 128, 4, 1, 0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=True),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        # One-hot encode labels
        batch_size = z.size(0)
        one_hot = torch.zeros(batch_size, self.num_classes, device=z.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Reshape for conv layers
        z = z.view(batch_size, self.latent_dim, 1, 1)
        one_hot = one_hot.view(batch_size, self.num_classes, 1, 1)
        
        # Concatenate
        x = torch.cat([z, one_hot], dim=1)
        return self.main(x)

# Conditional GAN - the complex one deployed
class ConditionalGAN(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(ConditionalGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Generator layers
        self.main = nn.Sequential(
            # Input: latent_dim + 50 (embedded label) = 150
            nn.ConvTranspose2d(latent_dim + 50, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 4x4x512
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8x8x256
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16x128
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32x32x64
            
            nn.ConvTranspose2d(64, 3, 1, 1, 0, bias=False),
            nn.Tanh()
            # 32x32x3 (CIFAR-10 resolution)
        )
    
    def forward(self, noise, labels):
        # Embed labels and reshape
        label_embed = self.label_embedding(labels).view(labels.size(0), 50, 1, 1)
        
        # Concatenate noise and embedded labels
        gen_input = torch.cat([noise, label_embed], 1)
        
        return self.main(gen_input)

def denormalize(tensor):
    """Convert from [-1, 1] to [0, 255]"""
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    return tensor.byte()

def save_images(images, prefix, output_dir='/root'):
    """Save images as PNG files"""
    for i, img in enumerate(images):
        # Convert to PIL Image
        img_array = img.cpu().permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img_array)
        
        # Save with descriptive name
        filename = f"{output_dir}/{prefix}_frog_{i+1}.png"
        pil_img.save(filename)
        print(f"Saved: {filename}")

def generate_tiny_frogs():
    """Generate frogs from TinyGenerator (the one with 79.5% accuracy)"""
    print("\n=== Generating from TinyGenerator (79.5% accuracy model) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load TinyGenerator
    generator = TinyGenerator(latent_dim=32, num_classes=10).to(device)
    state_dict = torch.load(
        '/root/proof_chain/cifar_gan_training/tiny_generator.pth',
        map_location=device,
        weights_only=False)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    generator.load_state_dict(state_dict
    ))
    generator.eval()
    
    # Generate 3 frog images (class 6)
    with torch.no_grad():
        z = torch.randn(3, 32).to(device)
        labels = torch.full((3,), 6, dtype=torch.long).to(device)  # Frog = class 6
        
        fake_images = generator(z, labels)
        fake_images = denormalize(fake_images)
    
    save_images(fake_images, "tinygen")
    print("✓ Generated 3 frog images from TinyGenerator")
    
    return fake_images

def generate_conditional_frogs():
    """Generate frogs from Conditional GAN (the deployed model with issues)"""
    print("\n=== Generating from Conditional GAN (deployed model) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Conditional GAN
    generator = ConditionalGAN(latent_dim=100, num_classes=10).to(device)
    
    # Load state dict and handle DataParallel
    state_dict = torch.load(
        '/root/proof_chain/cifar_gan_training/final_generator.pth',
        map_location=device,
        weights_only=False)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    generator.load_state_dict(state_dict
    )
    
    # Remove 'module.' prefix if present
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    generator.load_state_dict(state_dict)
    generator.eval()
    
    # Generate 3 frog images (class 6)
    with torch.no_grad():
        # Note: ConditionalGAN expects noise in specific shape
        noise = torch.randn(3, 100, 1, 1).to(device)
        labels = torch.full((3,), 6, dtype=torch.long).to(device)  # Frog = class 6
        
        fake_images = generator(noise, labels)
        fake_images = denormalize(fake_images)
    
    save_images(fake_images, "conditional")
    print("✓ Generated 3 frog images from Conditional GAN")
    
    return fake_images

def main():
    print("="*60)
    print("GENERATING FROG COMPARISON IMAGES")
    print("="*60)
    print("Generating 3 frog images from each GAN model")
    print("Frog is class 6 in CIFAR-10")
    
    # Generate from both models
    tiny_frogs = generate_tiny_frogs()
    cond_frogs = generate_conditional_frogs()
    
    print("\n" + "="*60)
    print("IMAGES SAVED TO /root/")
    print("="*60)
    print("TinyGenerator frogs (79.5% accuracy):")
    print("  - /root/tinygen_frog_1.png")
    print("  - /root/tinygen_frog_2.png")
    print("  - /root/tinygen_frog_3.png")
    print("\nConditional GAN frogs (deployed, 10% accuracy):")
    print("  - /root/conditional_frog_1.png")
    print("  - /root/conditional_frog_2.png")
    print("  - /root/conditional_frog_3.png")
    print("\n✓ All images generated successfully!")

if __name__ == '__main__':
    main()
