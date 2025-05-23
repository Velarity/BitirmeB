import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SkinCancerDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        # Etiket dosyasını oku
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Satırdan görsel yolu ve etiket bilgisi al
        image_path = self.data.loc[idx, 'full_path']
        label = self.data.loc[idx, 'label_encoded']

        # Görseli yükle
        image = Image.open(image_path).convert('RGB')

        # Dönüştürme işlemleri (varsa)
        if self.transform:
            image = self.transform(image)

        return image, label
