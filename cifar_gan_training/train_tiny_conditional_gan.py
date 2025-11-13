import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Hyperparameters - OPTIMIZED FOR ZK-ML
latent_dim = 32  # Reduced from 100
num_classes = 10
image_size = 32
channels = 3
batch_size = 128
num_epochs = 50
lr = 0.0002

class TinyGenerator(nn.Module):
    """Ultra-simplified Generator for ZK-ML compatibility"""
    def __init__(self, latent_dim, num_classes):
        super(TinyGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Input: [batch, latent_dim + num_classes, 1, 1] = [batch, 42, 1, 1]
        # One-hot class encoding is concatenated with latent vector in input

        # NO BATCHNORM - Major savings for ZK circuits!
        # Smaller channel dimensions: 128 -> 64 -> 32
        self.main = nn.Sequential(
            # Input: 42 x 1 x 1
            nn.ConvTranspose2d(latent_dim + num_classes, 128, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # 128 x 4 x 4

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # 64 x 8 x 8

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # 32 x 16 x 16

            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=True),
            nn.Tanh()
            # 3 x 32 x 32
        )

    def forward(self, z_with_class):
        # z_with_class is already [batch, latent_dim + num_classes, 1, 1]
        return self.main(z_with_class)

class TinyDiscriminator(nn.Module):
    """Ultra-simplified Discriminator"""
    def __init__(self, num_classes):
        super(TinyDiscriminator, self).__init__()
        self.num_classes = num_classes

        # NO BATCHNORM
        self.main = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(channels, 32, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 16 x 16

            nn.Conv2d(32, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 8 x 8

            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 4 x 4
        )

        # Separate outputs for real/fake and class prediction
        self.adv_layer = nn.Sequential(
            nn.Conv2d(128, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

        self.aux_layer = nn.Sequential(
            nn.Conv2d(128, num_classes, 4, 1, 0, bias=True)
        )

    def forward(self, img):
        features = self.main(img)
        validity = self.adv_layer(features).view(-1, 1)
        label = self.aux_layer(features).view(-1, num_classes)
        return validity, label

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'Available GPUs: {torch.cuda.device_count()}')

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize models
generator = TinyGenerator(latent_dim, num_classes).to(device)
discriminator = TinyDiscriminator(num_classes).to(device)

# Use DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

# Loss functions
adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Helper function to create one-hot encoded class labels
def create_input_with_class(batch_size, latent_dim, num_classes, labels, device):
    # Generate random latent vectors
    z = torch.randn(batch_size, latent_dim, 1, 1, device=device)

    # Create one-hot encoding for labels
    one_hot = torch.zeros(batch_size, num_classes, 1, 1, device=device)
    one_hot.scatter_(1, labels.view(-1, 1, 1, 1), 1)

    # Concatenate latent vector with one-hot class
    z_with_class = torch.cat([z, one_hot], dim=1)

    return z_with_class

# Training
print('Starting training...')
os.makedirs('checkpoints_tiny', exist_ok=True)

for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size_curr = imgs.size(0)
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Ground truths
        valid = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real images
        real_pred, real_aux = discriminator(imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) +
                      auxiliary_loss(real_aux, labels)) / 2

        # Fake images
        z_with_class = create_input_with_class(batch_size_curr, latent_dim, num_classes, labels, device)
        gen_imgs = generator(z_with_class)
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) +
                      auxiliary_loss(fake_aux, labels)) / 2

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        validity, pred_label = discriminator(gen_imgs)
        g_loss = (adversarial_loss(validity, valid) +
                 auxiliary_loss(pred_label, labels)) / 2

        g_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f'[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] '
                  f'[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]')

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, f'checkpoints_tiny/checkpoint_epoch_{epoch+1}.pth')
        print(f'Saved checkpoint at epoch {epoch+1}')

# Save final model
torch.save(generator.state_dict(), 'tiny_generator.pth')
torch.save(discriminator.state_dict(), 'tiny_discriminator.pth')

# Export to ONNX with single concatenated input
print('\nExporting to ONNX...')
generator.eval()

# Get the actual module if using DataParallel
gen_module = generator.module if isinstance(generator, nn.DataParallel) else generator

# Create example input: [1, 42, 1, 1] = latent (32) + one-hot class (10)
example_z = torch.randn(1, latent_dim, 1, 1)
example_class = torch.zeros(1, num_classes, 1, 1)
example_class[0, 3, 0, 0] = 1  # Class 3 (cat) as one-hot
example_input = torch.cat([example_z, example_class], dim=1)

print(f'Example input shape: {example_input.shape}')

torch.onnx.export(
    gen_module,
    example_input,
    'tiny_conditional_gan_cifar10.onnx',
    input_names=['latent_and_class'],
    output_names=['generated_image'],
    opset_version=12,
    do_constant_folding=True
)

print('✓ ONNX export complete: tiny_conditional_gan_cifar10.onnx')
print(f'\nModel size comparison:')
print(f'  Parameters: ~{sum(p.numel() for p in gen_module.parameters())/1e3:.1f}K')
print(f'  No BatchNorm layers')
print(f'  Latent dim: {latent_dim} (vs 100)')
print(f'  Max channels: 128 (vs 512)')
