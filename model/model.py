import torch
from torch import nn


model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2), # уменьшение в 2 раза
    
    nn.Conv2d(32, 64, kernel_size=64, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2), # еще

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((4, 4)), 

    nn.Flatten(), # 3d tensor -> 1d vector
    
    nn.Linear(256 * 4 * 4, 512), # автоопределение размера ленивое
    nn.ReLU(),
    nn.Dropout(0.40), # -40% neurons

    nn.Linear(512, 5)
)

if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"in {dummy_input.shape}")
    print(f"out {output.shape}")
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")