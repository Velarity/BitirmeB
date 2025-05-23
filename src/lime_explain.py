import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms

from model import SkinCancerCNN

# PyTorch modelimizi saran bir fonksiyon: LIME, numpy formatında çalışır
def predict_fn(images):
    model.eval()
    images = torch.tensor(images).permute(0, 3, 1, 2).float()  # N,H,W,C → N,C,H,W
    images = images / 255.0  # Normalizasyon
    images = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(images)
    with torch.no_grad():
        outputs = model(images.to(device))
        return torch.cat([1 - outputs, outputs], dim=1).cpu().numpy()

# Görseli oku ve orijinal haliyle döndür (numpy formatında)
def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))
    return np.array(image)

if __name__ == "__main__":
    # Modeli yükle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkinCancerCNN().to(device)
    model.load_state_dict(torch.load("outputs/model.pth", map_location=device))
    model.eval()

    # Görseli yükle
    image_path = "data/images/benign/0013.jpg"  # burayı değiştir
    image_np = load_image(image_path)

    # LIME açıklayıcısını başlat
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=1000
    )

    # Tahmin edilen sınıfı al
    pred_class = explanation.top_labels[0]

    # Görselleştir
    temp, mask = explanation.get_image_and_mask(
        label=pred_class,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(temp, mask))
    plt.title("LIME Explanation")
    plt.axis("off")
    plt.show()
