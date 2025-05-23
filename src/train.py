import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SkinCancerDataset
from model import SkinCancerCNN

# Cihaz (GPU varsa kullanılır)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dönüştürme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Verileri yükle
train_dataset = SkinCancerDataset(csv_path="data/train.csv", transform=transform)
test_dataset = SkinCancerDataset(csv_path="data/test.csv", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Modeli yükle
model = SkinCancerCNN().to(device)

# Kayıp fonksiyonu (binary classification için uygun)
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # boyut (B, 1) olmalı

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Doğruluk hesapla
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Aşama {epoch+1}/{num_epochs} - Kayıp: {avg_loss:.4f} - Doğruluk: {accuracy:.2f}%")

# Modeli kaydet
torch.save(model.state_dict(), "outputs/model.pth")
print("Eğitim tamamlandı ve model kaydedildi.")
