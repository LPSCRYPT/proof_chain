import torch
import torch.nn as nn

# Recreate the 32x32 generator architecture from the training script
class Generator32(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator32, self).__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(10, 50)
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + 50, 512, 4, 1, 0, bias=False),
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

# Typical 64x64 generator architecture
class Generator64(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator64, self).__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(10, 50)
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + 50, 512, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 1, 1, 0, bias=False),
            nn.Tanh()
        )

def count_params(model):
    return sum(p.numel() for p in model.parameters())

gen32 = Generator32()
gen64 = Generator64()

print('32x32 Generator Parameters:', count_params(gen32))
print('64x64 Generator Parameters:', count_params(gen64))
print('Parameter difference:', count_params(gen64) - count_params(gen32))
print('')
print('--- Computational Analysis ---')
print('32x32 output: 32 x 32 x 3 = 3,072 output values')
print('64x64 output: 64 x 64 x 3 = 12,288 output values')
print('Output size ratio:', round((64*64*3)/(32*32*3), 1), 'x more outputs for 64x64')
