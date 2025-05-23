import torch
import torch.nn as nn
import torch.nn.functional as F

class SkinCancerCNN(nn.Module):
    def __init__(self):
        super(SkinCancerCNN, self).__init__()
        
        # 1. Katman: Convolution + ReLU + MaxPooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Her seferinde boyutu yarıya indirir
        
        # 2. Katman: Convolution + ReLU + MaxPooling
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # 3. Katman: Convolution + ReLU + MaxPooling
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully connected layer (tam bağlantılı)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)  # Binary classification → 1 çıktı

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 64, 28, 28]
        x = x.reshape(-1, 64 * 28 * 28)          # Flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))        # 0-1 arası olasılık çıktısı
        return x
