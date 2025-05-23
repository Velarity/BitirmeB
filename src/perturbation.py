import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw
from model import SkinCancerCNN
from gradcam import preprocess_image

def mask_patch(image, i, j, patch_size):
    # Görselin (i,j) bölgesini karart (siyah dikdörtgen)
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(
        [j, i, j + patch_size, i + patch_size],
        fill=(0, 0, 0)
    )
    return img_copy

def generate_perturbation_map(image_path, patch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkinCancerCNN().to(device)
    model.load_state_dict(torch.load("outputs/model.pth", map_location=device))
    model.eval()

    # Görseli oku
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image_raw = Image.open(image_path).convert('RGB')
    input_tensor = transform(image_raw).unsqueeze(0).to(device)

    # Modelin orijinal tahminini al
    with torch.no_grad():
        original_pred = model(input_tensor).item()

    print(f"🔎 Orijinal tahmin (olasilik): {original_pred:.4f}")

    # Görseli parçala
    width, height = image_raw.size
    num_rows = height // patch_size
    num_cols = width // patch_size
    heatmap = np.zeros((num_rows, num_cols))

    for row in range(num_rows):
        for col in range(num_cols):
            i = row * patch_size
            j = col * patch_size

            perturbed_img = mask_patch(image_raw, i, j, patch_size)
            perturbed_tensor = transform(perturbed_img).unsqueeze(0).to(device)

            with torch.no_grad():
                perturbed_pred = model(perturbed_tensor).item()

            # Fark ne kadar büyükse o kadar önemli
            delta = abs(original_pred - perturbed_pred)
            heatmap[row, col] = delta

    # Normalize et
    heatmap /= np.max(heatmap)
    heatmap_resized = np.kron(heatmap, np.ones((patch_size, patch_size)))  # Patch'leri büyüt
    return heatmap_resized, image_raw

def show_heatmap_on_image(image, heatmap, alpha=0.5):
    heatmap_colored = np.uint8(255 * heatmap)
    heatmap_colored = np.stack([heatmap_colored]*3, axis=2)
    heatmap_colored = Image.fromarray(heatmap_colored).resize(image.size)
    heatmap_colored = np.array(heatmap_colored)

    image_np = np.array(image)
    overlay = np.uint8(np.clip(heatmap_colored * alpha + image_np, 0, 255))

    plt.imshow(overlay)
    plt.title("Perturbation-based Explanation")
    plt.axis("off")
    plt.show()

# Örnek kullanım
if __name__ == "__main__":
    image_path = "data/images/benign/0013.jpg"  # burayı uygun görselle değiştir
    heatmap, image = generate_perturbation_map(image_path, patch_size=32)
    plt.figure(1)
    show_heatmap_on_image(image, heatmap)
    plt.figure(2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Raw Perturbation Map")
    plt.colorbar()
    plt.axis("off")
    plt.show()
