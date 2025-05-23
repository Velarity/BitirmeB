from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SkinCancerDataset

# Dönüştürmeler (256x256 boyut, tensöre çevir, normalize et)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # RGB için
])

# Dataset oluştur
train_dataset = SkinCancerDataset(csv_path="data/train.csv", transform=transform)
test_dataset = SkinCancerDataset(csv_path="data/test.csv", transform=transform)

# DataLoader oluştur
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Test: İlk batch’i yazdır
for images, labels in train_loader:
    print(f"Görüntü şekli: {images.shape}")
    print(f"Etiketler: {labels}")
    break
