import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Hyperparameters
num_classes = 10
batch_size = 128
num_epochs = 20
lr = 0.001

class TinyClassifier(nn.Module):
    """Simplified CNN classifier for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(TinyClassifier, self).__init__()

        # No BatchNorm for ZK compatibility
        self.features = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 32, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 32 x 16 x 16

            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 64 x 8 x 8

            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Data loading
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

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model
classifier = TinyClassifier(num_classes).to(device)

# Use DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs')
    classifier = nn.DataParallel(classifier)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training
print('Starting training...')
os.makedirs('checkpoints_classifier', exist_ok=True)

best_acc = 0.0

for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward pass
        outputs = classifier(imgs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 100 == 0:
            print(f'[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')

    scheduler.step()

    # Validation
    classifier.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = classifier(imgs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f'Epoch {epoch+1} - Test Accuracy: {test_acc:.2f}%')

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': test_acc,
        }, 'checkpoints_classifier/best_classifier.pth')
        print(f'Saved best model with accuracy: {test_acc:.2f}%')

# Save final model
torch.save(classifier.state_dict(), 'tiny_classifier.pth')

# Export to ONNX
print('\nExporting to ONNX...')
classifier.eval()

# Get the actual module if using DataParallel
cls_module = classifier.module if isinstance(classifier, nn.DataParallel) else classifier

# Create example input
example_input = torch.randn(1, 3, 32, 32).to(device)

torch.onnx.export(
    cls_module,
    example_input,
    'tiny_classifier_cifar10.onnx',
    input_names=['image'],
    output_names=['class_logits'],
    opset_version=12,
    do_constant_folding=True
)

print('✓ ONNX export complete: tiny_classifier_cifar10.onnx')
print(f'\nFinal Test Accuracy: {best_acc:.2f}%')
print(f'Parameters: ~{sum(p.numel() for p in cls_module.parameters())/1e3:.1f}K')
