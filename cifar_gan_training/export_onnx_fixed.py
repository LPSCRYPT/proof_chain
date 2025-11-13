import torch
import torch.nn as nn

# Hyperparameters
latent_dim = 32
num_classes = 10
image_size = 32
channels = 3

class TinyGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes):
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
            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=True),
            nn.Tanh()
        )
    
    def forward(self, z_with_class):
        return self.main(z_with_class)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the trained generator
generator = TinyGenerator(latent_dim, num_classes).to(device)

# Use DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator)

# Load the final checkpoint
checkpoint = torch.load('tiny_generator.pth')
generator.load_state_dict(checkpoint)
generator.eval()

# Get the actual module if using DataParallel
gen_module = generator.module if isinstance(generator, nn.DataParallel) else generator

print('Exporting to ONNX...')

# Create example input on the SAME DEVICE as model
example_z = torch.randn(1, latent_dim, 1, 1).to(device)
example_class = torch.zeros(1, num_classes, 1, 1).to(device)
example_class[0, 3, 0, 0] = 1  # Class 3 (cat) as one-hot
example_input = torch.cat([example_z, example_class], dim=1)

print(f'Example input shape: {example_input.shape}')
print(f'Example input device: {example_input.device}')

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
print(f'Model size comparison:')
print(f'  Parameters: ~{sum(p.numel() for p in gen_module.parameters())/1e3:.1f}K')
print(f'  No BatchNorm layers')
print(f'  Latent dim: {latent_dim} (vs 100)')
print(f'  Max channels: 128 (vs 512)')
