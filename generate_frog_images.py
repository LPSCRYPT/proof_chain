#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# TinyGenerator
class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(42, 128, 4, 1, 0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=True),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        batch_size = z.size(0)
        one_hot = torch.zeros(batch_size, 10, device=z.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        z = z.view(batch_size, 32, 1, 1)
        one_hot = one_hot.view(batch_size, 10, 1, 1)
        x = torch.cat([z, one_hot], dim=1)
        return self.main(x)

# Conditional GAN
class ConditionalGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embedding = nn.Embedding(10, 50)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(150, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 1, 1, 0, bias=False),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels).view(labels.size(0), 50, 1, 1)
        gen_input = torch.cat([noise, label_embed], 1)
        return self.main(gen_input)

def save_frog_images():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Generating frog images from both GANs...')
    
    # TinyGenerator
    print('\n1. TinyGenerator (79.5% accuracy):')
    gen1 = TinyGenerator().to(device)
    state1 = torch.load('/root/proof_chain/cifar_gan_training/tiny_generator.pth', map_location=device, weights_only=False)
    if list(state1.keys())[0].startswith('module.'):
        state1 = {k[7:]: v for k, v in state1.items()}
    gen1.load_state_dict(state1)
    gen1.eval()
    
    with torch.no_grad():
        z = torch.randn(3, 32).to(device)
        labels = torch.full((3,), 6, dtype=torch.long).to(device)
        imgs1 = gen1(z, labels)
        imgs1 = ((imgs1 + 1) / 2 * 255).byte()
        
    for i in range(3):
        img = imgs1[i].cpu().permute(1, 2, 0).numpy()
        Image.fromarray(img).save(f'/root/tinygen_frog_{i+1}.png')
        print(f'   Saved: /root/tinygen_frog_{i+1}.png')
    
    # Conditional GAN
    print('\n2. Conditional GAN (deployed, 10% accuracy):')
    gen2 = ConditionalGAN().to(device)
    state2 = torch.load('/root/proof_chain/cifar_gan_training/final_generator.pth', map_location=device, weights_only=False)
    if list(state2.keys())[0].startswith('module.'):
        state2 = {k[7:]: v for k, v in state2.items()}
    gen2.load_state_dict(state2)
    gen2.eval()
    
    with torch.no_grad():
        noise = torch.randn(3, 100, 1, 1).to(device)
        labels = torch.full((3,), 6, dtype=torch.long).to(device)
        imgs2 = gen2(noise, labels)
        imgs2 = ((imgs2 + 1) / 2 * 255).byte()
        
    for i in range(3):
        img = imgs2[i].cpu().permute(1, 2, 0).numpy()
        Image.fromarray(img).save(f'/root/conditional_frog_{i+1}.png')
        print(f'   Saved: /root/conditional_frog_{i+1}.png')
    
    print('\nDone! All 6 frog images saved to /root/')

save_frog_images()
