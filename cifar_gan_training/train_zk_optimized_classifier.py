#!/usr/bin/env python3
"""
ZK-Optimized CIFAR-10 Classifier
Designed to minimize verifier size by avoiding MaxPool operations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time
import argparse

class ZKOptimizedClassifierV1(nn.Module):
    """Version 1: Uses AvgPool instead of MaxPool (linear operation, ZK-friendly)"""
    def __init__(self, num_classes=10):
        super(ZKOptimizedClassifierV1, self).__init__()
        
        self.features = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 32, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),  # AvgPool is linear, much cheaper than MaxPool!
            # 32 x 16 x 16
            
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),
            # 64 x 8 x 8
            
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),
            # 128 x 4 x 4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_classes, bias=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ZKOptimizedClassifierV2(nn.Module):
    """Version 2: Uses strided convolutions instead of pooling"""
    def __init__(self, num_classes=10):
        super(ZKOptimizedClassifierV2, self).__init__()
        
        self.features = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=True),  # Stride=2 replaces pooling
            nn.ReLU(True),
            # 32 x 16 x 16
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            # 64 x 8 x 8
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            # 128 x 4 x 4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_classes, bias=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ZKOptimizedClassifierV3(nn.Module):
    """Version 3: Fully convolutional with global average pooling"""
    def __init__(self, num_classes=10):
        super(ZKOptimizedClassifierV3, self).__init__()
        
        self.features = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=True),  # Downsample with conv
            nn.ReLU(True),
            # 32 x 16 x 16
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            # 64 x 8 x 8
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            # 128 x 4 x 4
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            # 256 x 4 x 4
        )
        
        # Global average pooling + linear (very ZK-friendly)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 256 x 1 x 1
        self.classifier = nn.Linear(256, num_classes, bias=True)
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_model(model, device, train_loader, test_loader, epochs=20, lr=0.001):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    model.train()
    best_acc = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 99:
                print(f'[Epoch {epoch+1}, Batch {batch_idx+1}] Loss: {running_loss/100:.3f}')
                running_loss = 0.0
        
        train_acc = 100. * correct / total
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        model.train()
        scheduler.step()
    
    return best_acc


def export_to_onnx(model, model_path, dummy_input):
    """Export model to ONNX format"""
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ZK-Optimized CIFAR-10 Classifier')
    parser.add_argument('--version', type=int, default=1, choices=[1, 2, 3],
                        help='Model version: 1=AvgPool, 2=StridedConv, 3=GlobalAvgPool')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    # Data preparation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                           shuffle=False, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Select model version
    if args.version == 1:
        model = ZKOptimizedClassifierV1()
        model_name = 'zk_classifier_avgpool'
        print("Using V1: AvgPool version")
    elif args.version == 2:
        model = ZKOptimizedClassifierV2()
        model_name = 'zk_classifier_strided'
        print("Using V2: Strided convolution version")
    else:
        model = ZKOptimizedClassifierV3()
        model_name = 'zk_classifier_global'
        print("Using V3: Global average pooling version")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    best_acc = train_model(model, device, trainloader, testloader, 
                          epochs=args.epochs, lr=args.lr)
    print(f"\nBest test accuracy: {best_acc:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), f'{model_name}.pth')
    print(f"Model saved to {model_name}.pth")
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    export_to_onnx(model, f'{model_name}.onnx', dummy_input)
    
    # Test single inference
    model.eval()
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"\nTest output shape: {test_output.shape}")
        print(f"Test output: {test_output[0].cpu().numpy()}")


if __name__ == '__main__':
    main()
