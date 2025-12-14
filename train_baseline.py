import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ============================================================================
# MOBILENET-V2 IMPLEMENTATION (Original kuangliu/pytorch-cifar)
# ============================================================================

class Block(nn.Module):
    """MobileNetV2 Inverted Residual Block"""
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2(nn.Module):
    """MobileNetV2 architecture for CIFAR-10"""
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
    
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
    
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data():
    """Prepare CIFAR-10 dataset with transforms"""
    print("Preparing CIFAR-10 dataset...")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    print(f"Training set: {len(trainset)} images")
    print(f"Test set: {len(testset)} images")
    
    return trainloader, testloader

# ============================================================================
# TRAINING AND TESTING FUNCTIONS
# ============================================================================

def train_epoch(net, device, trainloader, optimizer, criterion, epoch):
    """Train model for one epoch"""
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch {epoch} | Batch {batch_idx+1}/{len(trainloader)} | '
                  f'Loss: {running_loss/(batch_idx+1):.3f} | '
                  f'Acc: {100.*correct/total:.2f}%')
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    print(f'Epoch {epoch} | Loss: {train_loss:.3f} | Acc: {train_acc:.3f}%')
    
    return train_loss, train_acc

def test_model(net, device, testloader, criterion):
    """Test model on test set"""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
    
    return test_loss, test_acc

def plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies):
    """Plot training and validation curves"""
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Loss Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    plt.title('Training and Test Loss vs. Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    plt.title('Training and Test Accuracy vs. Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function"""
    print("\n" + "="*80)
    print("BASELINE MODEL TRAINING (FP32)")
    print("="*80)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Prepare data
    trainloader, testloader = prepare_data()
    
    # Initialize model (original MobileNetV2 for baseline FP32 training)
    print("\nInitializing MobileNetV2 model for FP32 baseline...")
    net = MobileNetV2(num_classes=10).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size (FP32): {total_params * 4 / (1024*1024):.2f} MB")
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Initialize variables for best model tracking
    best_acc = 0.0
    best_epoch = 0
    
    # Training loop
    print(f"\nStarting training for 50 epochs...")
    print("-" * 80)
    
    for epoch in range(1, 51):  # 50 epochs
        current_train_loss, current_train_acc = train_epoch(net, device, trainloader, optimizer, criterion, epoch)
        current_test_loss, current_test_acc = test_model(net, device, testloader, criterion)
        
        train_losses.append(current_train_loss)
        train_accuracies.append(current_train_acc)
        test_losses.append(current_test_loss)
        test_accuracies.append(current_test_acc)
        
        scheduler.step()
        
        # Save best model
        if current_test_acc > best_acc:
            best_acc = current_test_acc
            best_epoch = epoch
            torch.save(net.state_dict(), 'best_baseline_model.pth')
            print(f'✓ Saved best model with test accuracy: {best_acc:.3f}% at epoch {best_epoch}')
        
        print("-" * 80)
    
    # Training completed
    print('Training complete!')
    print(f'Best Test Accuracy: {best_acc:.3f}% achieved at Epoch {best_epoch}')
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Model: MobileNetV2 (FP32 baseline)")
    print(f"Dataset: CIFAR-10")
    print(f"Training epochs: 50")
    print(f"Best test accuracy: {best_acc:.3f}% (epoch {best_epoch})")
    print(f"Final test accuracy: {test_accuracies[-1]:.3f}%")
    print(f"Model saved as: best_baseline_model.pth")
    print(f"Training curves saved as: baseline_training_curves.png")
    print("="*80)

if __name__ == '__main__':
    main()