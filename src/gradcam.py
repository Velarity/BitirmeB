import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import SkinCancerCNN

# Grad-CAM için yardımcı sınıf
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook’lar: Geri yayılım ve ileri yayılım sırasında veri toplamak için
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        class_idx = torch.argmax(output).item()

        # Geri yayılımı başlat
        self.model.zero_grad()
        output[0][0].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu()
        heatmap = torch.clamp(heatmap, min=0)  # Negatifleri sıfırla
        if torch.max(heatmap) != 0:
             heatmap /= torch.max(heatmap)  # Normalize et ama sıfıra bölme hatasından kaçın

        return heatmap.numpy()

# Görseli yükle ve hazırla
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image

# Grad-CAM görselleştirme fonksiyonu

def show_gradcam_on_image(pil_img, heatmap, alpha=0.5, save_path=None):
    # Görsel boyutuna göre heatmap'i yeniden boyutlandır
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(pil_img.size)
    heatmap_resized = np.array(heatmap_resized)

    # Heatmap'i 3 kanallı hale getir (RGB ile eşleşsin)
    if heatmap_resized.ndim == 2:
        heatmap_resized = np.stack([heatmap_resized]*3, axis=2)  # (H, W) → (H, W, 3)

    # Orijinal görseli numpy formatına çevir
    original_img = np.array(pil_img)

    # Görsellerin boyutlarının gerçekten eşleştiğinden emin ol
    assert heatmap_resized.shape == original_img.shape, f"Shape mismatch: {heatmap_resized.shape} vs {original_img.shape}"

    # Isı haritasını bindir
    superimposed_img = heatmap_resized * alpha + original_img
    superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))

    # Göster veya kaydet
    plt.imshow(superimposed_img)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📸 Grad-CAM kaydedildi: {save_path}")
    else:
        plt.show()



# Ana çalışma fonksiyonu
def run_gradcam(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkinCancerCNN().to(device)
    model.load_state_dict(torch.load("outputs/model.pth", map_location=device))

    input_tensor, raw_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    gradcam = GradCAM(model, target_layer=model.conv3)
    heatmap = gradcam.generate(input_tensor)
    plt.figure(1)
    show_gradcam_on_image(raw_image, heatmap, alpha=0.7)

    # 🔥 2️⃣ Sadece heatmap (renkli)
    plt.figure(2)
    plt.imshow(heatmap, cmap='jet')  # Isı haritasını renkli olarak tek başına göster
    plt.colorbar()
    plt.title("Grad-CAM Heatmap (Raw)")
    plt.axis("off")
    plt.show()

    show_gradcam_on_image(raw_image, heatmap, alpha=0.7)

# Örnek kullanım
if __name__ == "__main__":
    # İstediğin test görselinin yolunu buraya yaz (örnek: benign/ISIC_0001.jpg)
    sample_path = "data/images/benign/0019.jpg"
    
    run_gradcam(sample_path)
