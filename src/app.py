import os
import torch
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import SkinCancerCNN
from gradcam import GradCAM, preprocess_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Modeli yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCancerCNN().to(device)
model.load_state_dict(torch.load("outputs/model.pth", map_location=device))
model.eval()

# Sayfa başlığı
st.title("Deri Kanseri Sınıflandırması ve Açıklanabilirlik")

# Görsel yükleme
uploaded_file = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # Görseli modele uygun tensor haline getir
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor).item()
        label = "Malignant (Kötü Huylu)" if prediction > 0.5 else "Benign (İyi Huylu)"
        st.write(f"**Tahmin:** {label} ({prediction:.2f})")

    # Grad-CAM
    st.subheader("Grad-CAM Açıklaması")
    gradcam = GradCAM(model, target_layer=model.conv3)
    heatmap = gradcam.generate(input_tensor)
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size)
    heatmap = np.array(heatmap)

    image_np = np.array(image)
    overlay = np.uint8(heatmap * 0.5 + image_np)
    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

    # LIME
    st.subheader("LIME Açıklaması")
    explainer = lime_image.LimeImageExplainer()
    image_for_lime = np.array(image)

    def predict_fn(images):
        images = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0
        images = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(images)
        with torch.no_grad():
            outputs = model(images.to(device))
            return torch.cat([1 - outputs, outputs], dim=1).cpu().numpy()

    explanation = explainer.explain_instance(
        image_for_lime,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    lime_img, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    st.image(mark_boundaries(lime_img, mask), caption="LIME Explanation", use_container_width=True)
