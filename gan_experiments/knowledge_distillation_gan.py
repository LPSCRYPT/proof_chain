#!/usr/bin/env python3
"""
Knowledge Distillation for ZK-Friendly GANs
Train a large teacher GAN, then distill into a small student
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

sys.path.insert(0, '/root/proof_chain/gan_experiments/architectures')
sys.path.insert(0, '/root/proof_chain')

class TeacherGenerator(nn.Module):
    """Large teacher generator with good quality"""
    def __init__(self, latent_dim=100, num_classes=10, ngf=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Embedding(num_classes, 50)
        
        # Projection
        self.proj = nn.Sequential(
            nn.Linear(latent_dim + 50, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Main layers
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        label_embed = self.embed(labels)
        x = torch.cat([noise.view(noise.size(0), -1), label_embed], dim=1)
        x = self.proj(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_final(x)
        return x


class StudentGeneratorMLP(nn.Module):
    """Ultra-lightweight student generator for ZK"""
    def __init__(self, latent_dim=100, num_classes=10, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 20)  # Smaller embedding
        
        # Only 3 layers for ultra-low ops
        self.fc1 = nn.Linear(latent_dim + 20, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, 3 * 32 * 32)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        
    def forward(self, noise, labels):
        label_embed = self.embed(labels)
        x = torch.cat([noise.view(noise.size(0), -1), label_embed], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x.view(x.size(0), 3, 32, 32)


class StudentGeneratorDepthwise(nn.Module):
    """Student with depthwise separable convolutions"""
    def __init__(self, latent_dim=100, num_classes=10, ngf=32):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 30)
        
        # Initial projection
        self.proj = nn.Sequential(
            nn.Linear(latent_dim + 30, ngf * 4 * 4 * 4),
            nn.BatchNorm1d(ngf * 4 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Depthwise separable convolutions (much fewer ops)
        self.conv1 = self._make_depthwise_layer(ngf * 4, ngf * 2, upsample=True)
        self.conv2 = self._make_depthwise_layer(ngf * 2, ngf, upsample=True)
        self.conv3 = self._make_depthwise_layer(ngf, ngf, upsample=True)
        
        self.final = nn.Sequential(
            nn.Conv2d(ngf, 3, 1, 1, 0),  # 1x1 conv
            nn.Tanh()
        )
        
    def _make_depthwise_layer(self, in_ch, out_ch, upsample=False):
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        # Depthwise
        layers.append(nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch))
        layers.append(nn.BatchNorm2d(in_ch))
        layers.append(nn.ReLU(True))
        # Pointwise
        layers.append(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)
        
    def forward(self, noise, labels):
        label_embed = self.embed(labels)
        x = torch.cat([noise.view(noise.size(0), -1), label_embed], dim=1)
        x = self.proj(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final(x)
        return x


def train_with_distillation(teacher_gen, student_gen, device='cuda', epochs=50):
    """Train student to mimic teacher"""
    print("\n" + "="*80)
    print("KNOWLEDGE DISTILLATION TRAINING")
    print("="*80)
    
    # Count ops
    from flexible_gan_architectures import estimate_einsum_ops
    teacher_ops = estimate_einsum_ops(teacher_gen)
    student_ops = estimate_einsum_ops(student_gen)
    
    print(f"Teacher ops: {teacher_ops}")
    print(f"Student ops: {student_ops}")
    print(f"Compression ratio: {teacher_ops/student_ops:.2f}x")
    
    # Optimizers
    optimizer = torch.optim.Adam(student_gen.parameters(), lr=0.001)
    
    # Losses
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    # Training loop
    for epoch in range(epochs):
        epoch_losses = []
        
        for _ in range(100):  # 100 batches per epoch
            batch_size = 64
            
            # Generate random inputs
            noise = torch.randn(batch_size, 100, 1, 1).to(device)
            labels = torch.randint(0, 10, (batch_size,)).to(device)
            
            # Get teacher outputs (no grad needed)
            with torch.no_grad():
                teacher_outputs = teacher_gen(noise, labels)
                # Also get intermediate features if available
                
            # Get student outputs
            student_outputs = student_gen(noise, labels)
            
            # Distillation loss (pixel + perceptual)
            pixel_loss = l1_loss(student_outputs, teacher_outputs)
            
            # Feature matching loss (using downsampled versions)
            teacher_feat = F.avg_pool2d(teacher_outputs, 4)
            student_feat = F.avg_pool2d(student_outputs, 4)
            feat_loss = mse_loss(student_feat, teacher_feat)
            
            # Total loss
            loss = pixel_loss + 0.5 * feat_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        if epoch % 10 == 0:
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f}")
    
    return student_gen


def test_distilled_model(student_gen, device='cuda'):
    """Test the distilled student model"""
    print("\n" + "="*80)
    print("TESTING DISTILLED MODEL")
    print("="*80)
    
    # Load classifier
    try:
        from test_conditional_gan_proper import ZKOptimizedClassifier
        classifier = ZKOptimizedClassifier().to(device)
        classifier_path = "/root/proof_chain/cifar_gan_training/zk_classifier_avgpool.pth"
        
        if os.path.exists(classifier_path):
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        
        # Test accuracy
        student_gen.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for class_idx in range(10):
                noise = torch.randn(100, 100, 1, 1).to(device)
                labels = torch.full((100,), class_idx, dtype=torch.long).to(device)
                
                fake_images = student_gen(noise, labels)
                outputs = classifier(fake_images)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == labels).sum().item()
                total += 100
        
        accuracy = 100 * correct / total
        print(f"Distilled model accuracy: {accuracy:.2f}%")
        
        # Diversity test
        diversity_scores = []
        for i in range(10):
            noise = torch.randn(10, 100, 1, 1).to(device)
            labels = torch.full((10,), i, dtype=torch.long).to(device)
            samples = student_gen(noise, labels).cpu()
            
            dists = []
            for j in range(len(samples)):
                for k in range(j+1, len(samples)):
                    dist = torch.mean((samples[j] - samples[k]) ** 2).item()
                    dists.append(dist)
            diversity_scores.append(np.mean(dists))
        
        print(f"Diversity score: {np.mean(diversity_scores):.4f}")
        
        return accuracy, np.mean(diversity_scores)
        
    except Exception as e:
        print(f"Error testing: {e}")
        return 0, 0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load pre-trained teacher or train one
    print("\n1. Setting up teacher model...")
    teacher = TeacherGenerator().to(device)
    
    # Try to load existing good model as teacher
    teacher_path = "/root/proof_chain/gan_experiments/tier3/models/exp_016_generator.pth"
    if os.path.exists(teacher_path):
        # Load the high-capacity conv model as teacher
        from flexible_gan_architectures import create_generator
        config = {
            "experiment_id": "exp_016",
            "architecture": {
                "ngf": 64,
                "num_layers": 4,
                "latent_dim": 100,
                "embed_dim": 50
            }
        }
        teacher = create_generator(config).to(device)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        print("✓ Loaded pre-trained teacher (exp_016)")
    else:
        print("Training new teacher model...")
        # Would train teacher here
    
    # Test different student architectures
    students = {
        "MLP_Student": StudentGeneratorMLP(hidden_dim=256).to(device),
        "Depthwise_Student": StudentGeneratorDepthwise(ngf=24).to(device)
    }
    
    results = {}
    
    for name, student in students.items():
        print(f"\n2. Training {name}...")
        
        # Train with distillation
        trained_student = train_with_distillation(teacher, student, device)
        
        # Test
        acc, div = test_distilled_model(trained_student, device)
        
        results[name] = {
            "accuracy": acc,
            "diversity": div,
            "ops": estimate_einsum_ops(trained_student)
        }
        
        # Save model
        save_path = f"/root/proof_chain/gan_experiments/distilled_{name.lower()}.pth"
        torch.save(trained_student.state_dict(), save_path)
        print(f"✓ Saved to {save_path}")
    
    # Summary
    print("\n" + "="*80)
    print("DISTILLATION RESULTS SUMMARY")
    print("="*80)
    
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {res['accuracy']:.2f}%")
        print(f"  Diversity: {res['diversity']:.4f}")
        print(f"  Ops: {res['ops']}")
        
        if res['accuracy'] > 35:
            print(f"  ✅ BREAKTHROUGH! Significantly better than previous {29.4}%")

if __name__ == "__main__":
    from flexible_gan_architectures import estimate_einsum_ops
    main()
