import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# ============================================================================
# QUANTIZATION MODULES (From notebook implementation)
# ============================================================================

class FakeQuantize(nn.Module):
    """Fake quantization module that simulates quantization during training"""
    def __init__(self, num_bits=8, symmetric=True):
        super(FakeQuantize, self).__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0))
        
        if symmetric:
            self.qmin = -2**(num_bits - 1) + 1
            self.qmax = 2**(num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**num_bits - 1
    
    def forward(self, x):
        if self.training:
            # Update scale based on current batch statistics
            x_min = x.min().item()
            x_max = x.max().item()
            if self.symmetric:
                max_val = max(abs(x_min), abs(x_max))
                scale = max_val / self.qmax if max_val > 0 else 1e-8
                zero_point = 0
            else:
                scale = (x_max - x_min) / (self.qmax - self.qmin) if x_max != x_min else 1e-8
                zero_point = self.qmin - round(x_min / scale)
            
            self.scale.fill_(scale)
            self.zero_point.fill_(zero_point)
        
        # Simulate quantization
        x_quant = torch.clamp(torch.round(x / self.scale) + self.zero_point, 
                              self.qmin, self.qmax)
        # Dequantize back to FP32
        x_dequant = (x_quant - self.zero_point) * self.scale
        
        return x_dequant

class QuantStub(nn.Module):
    """Stub module to mark the start of quantized region"""
    def __init__(self, num_bits=8):
        super(QuantStub, self).__init__()
        self.fake_quant = FakeQuantize(num_bits=num_bits, symmetric=True)
    
    def forward(self, x):
        return self.fake_quant(x)

class DeQuantStub(nn.Module):
    """Stub module to mark the end of quantized region"""
    def __init__(self):
        super(DeQuantStub, self).__init__()
    
    def forward(self, x):
        return x

# ============================================================================
# QUANTIZED MOBILENET-V2 IMPLEMENTATION
# ============================================================================

class QuantizedBlock(nn.Module):
    """MobileNetV2 Inverted Residual Block with Fake Quantization"""
    def __init__(self, in_planes, out_planes, expansion, stride, quantize=False):
        super(QuantizedBlock, self).__init__()
        self.stride = stride
        self.quantize = quantize
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
        
        # Add fake quantization for activations if enabled
        if quantize:
            self.quant_act1 = FakeQuantize(num_bits=8, symmetric=True)
            self.quant_act2 = FakeQuantize(num_bits=8, symmetric=True)
            self.quant_act3 = FakeQuantize(num_bits=8, symmetric=True)
            # Quantize weights
            self.weight_quant1 = FakeQuantize(num_bits=8, symmetric=True)
            self.weight_quant2 = FakeQuantize(num_bits=8, symmetric=True)
            self.weight_quant3 = FakeQuantize(num_bits=8, symmetric=True)
    
    def forward(self, x):
        if self.quantize:
            # Quantize conv1 weights and activations
            weight1_orig = self.conv1.weight.data.clone()
            self.conv1.weight.data = self.weight_quant1(self.conv1.weight.data)
            out = F.relu(self.bn1(self.conv1(x)))
            self.conv1.weight.data = weight1_orig
            out = self.quant_act1(out)
            
            # Quantize conv2 weights and activations
            weight2_orig = self.conv2.weight.data.clone()
            self.conv2.weight.data = self.weight_quant2(self.conv2.weight.data)
            out = F.relu(self.bn2(self.conv2(out)))
            self.conv2.weight.data = weight2_orig
            out = self.quant_act2(out)
            
            # Quantize conv3 weights
            weight3_orig = self.conv3.weight.data.clone()
            self.conv3.weight.data = self.weight_quant3(self.conv3.weight.data)
            out = self.bn3(self.conv3(out))
            self.conv3.weight.data = weight3_orig
            out = self.quant_act3(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class QuantizedMobileNetV2(nn.Module):
    """MobileNetV2 with configurable quantization support"""
    cfg = [(1,  16, 1, 1),   # expansion, out_planes, num_blocks, stride
           (6,  24, 2, 1),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
    
    def __init__(self, num_classes=10, quantize=False):
        super(QuantizedMobileNetV2, self).__init__()
        self.quantize = quantize
        
        # Input quantization stub
        if quantize:
            self.quant = QuantStub(num_bits=8)
        
        # First convolution layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Inverted residual blocks
        self.layers = self._make_layers(in_planes=32, quantize=quantize)
        
        # Final convolution
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Classifier
        self.linear = nn.Linear(1280, num_classes)
        
        # Add fake quantization modules if enabled
        if quantize:
            self.quant_conv1_weight = FakeQuantize(num_bits=8, symmetric=True)
            self.quant_conv1_act = FakeQuantize(num_bits=8, symmetric=True)
            self.quant_conv2_weight = FakeQuantize(num_bits=8, symmetric=True)
            self.quant_conv2_act = FakeQuantize(num_bits=8, symmetric=True)
            self.quant_linear_weight = FakeQuantize(num_bits=8, symmetric=True)
            self.dequant = DeQuantStub()
    
    def _make_layers(self, in_planes, quantize=False):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(QuantizedBlock(in_planes, out_planes, expansion, stride, quantize=quantize))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input quantization
        if self.quantize:
            x = self.quant(x)
        
        # First conv with quantization
        if self.quantize:
            weight_orig = self.conv1.weight.data.clone()
            self.conv1.weight.data = self.quant_conv1_weight(self.conv1.weight.data)
            out = F.relu(self.bn1(self.conv1(x)))
            self.conv1.weight.data = weight_orig
            out = self.quant_conv1_act(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        
        # Inverted residual blocks
        out = self.layers(out)
        
        # Final conv with quantization
        if self.quantize:
            weight_orig = self.conv2.weight.data.clone()
            self.conv2.weight.data = self.quant_conv2_weight(self.conv2.weight.data)
            out = F.relu(self.bn2(self.conv2(out)))
            self.conv2.weight.data = weight_orig
            out = self.quant_conv2_act(out)
        else:
            out = F.relu(self.bn2(self.conv2(out)))
        
        # Global average pooling
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        # Linear layer with quantization
        if self.quantize:
            weight_orig = self.linear.weight.data.clone()
            self.linear.weight.data = self.quant_linear_weight(self.linear.weight.data)
            out = self.linear(out)
            self.linear.weight.data = weight_orig
            out = self.dequant(out)
        else:
            out = self.linear(out)
        
        return out

# Backward compatibility alias
MobileNetV2 = QuantizedMobileNetV2

def test_model(model, device, testloader, criterion):
    """Test the model and return accuracy and loss"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / (batch_idx + 1)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Prepare test dataset
    print('Preparing CIFAR-10 test dataset...')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=100, 
        shuffle=False, 
        num_workers=2
    )
    
    # Load model
    print('Loading QuantizedMobileNetV2 model...')
    net = QuantizedMobileNetV2(num_classes=10, quantize=True).to(device)
    
    # Load pretrained weights
    print('Loading QAT retrained weights from best_model_qat_retrained.pth...')
    try:
        checkpoint = torch.load('best_model_qat_retrained.pth', map_location=device)
        net.load_state_dict(checkpoint, strict=False)  # Use strict=False to handle any missing keys
        print("QAT model weights loaded successfully!")
    except FileNotFoundError:
        print("Error: best_model_qat_retrained.pth not found!")
        print("Please run the QAT training in the notebook first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Define loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Test the model
    print('\nTesting the model...')
    test_loss, test_acc = test_model(net, device, testloader, criterion)
    
    # Print results
    print('\n' + '='*50)
    print('QAT RETRAINED MODEL TEST RESULTS')
    print('='*50)
    print(f'Device: {device}')
    print(f'Model: QuantizedMobileNetV2 (quantize=True)')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    print('='*50)
    print('Note: This model uses fake quantization simulation.')
    print('For true INT8 deployment, convert using PyTorch quantization APIs.')
    print('='*50)

if __name__ == '__main__':
    main()
