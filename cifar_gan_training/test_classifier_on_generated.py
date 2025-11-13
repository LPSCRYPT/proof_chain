import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob

class TinyClassifier(nn.Module):
    """Simplified CNN classifier for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(TinyClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
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

# Load classifier
classifier = TinyClassifier(num_classes=10).to(device)

if torch.cuda.device_count() > 1:
    classifier = nn.DataParallel(classifier)

checkpoint = torch.load('checkpoints_classifier/best_classifier.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()

print(f"Loaded classifier with test accuracy: {checkpoint['accuracy']:.2f}%")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Transform for inference
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test on generated images
generated_dir = 'generated_images'
results = {class_name: {'correct': 0, 'total': 0} for class_name in class_names}

print('\nClassifying generated images...\n')

for true_class_idx, true_class_name in enumerate(class_names):
    pattern = f'{generated_dir}/{true_class_name}_*.png'
    image_files = glob.glob(pattern)

    for img_path in sorted(image_files):
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Classify
        with torch.no_grad():
            output = classifier(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class_idx = output.argmax(1).item()
            pred_class_name = class_names[pred_class_idx]
            confidence = probabilities[0, pred_class_idx].item() * 100

        # Track results
        results[true_class_name]['total'] += 1
        if pred_class_idx == true_class_idx:
            results[true_class_name]['correct'] += 1

        # Print result
        status = '✓' if pred_class_idx == true_class_idx else '✗'
        print(f'{status} {os.path.basename(img_path):20s} -> '
              f'Predicted: {pred_class_name:10s} (confidence: {confidence:5.1f}%)')

# Summary
print('\n' + '='*60)
print('SUMMARY')
print('='*60)

total_correct = 0
total_images = 0

for class_name in class_names:
    correct = results[class_name]['correct']
    total = results[class_name]['total']
    accuracy = 100. * correct / total if total > 0 else 0
    print(f'{class_name:12s}: {correct:2d}/{total:2d} correct ({accuracy:5.1f}%)')
    total_correct += correct
    total_images += total

overall_acc = 100. * total_correct / total_images if total_images > 0 else 0
print('='*60)
print(f'Overall: {total_correct}/{total_images} correct ({overall_acc:.1f}%)')
