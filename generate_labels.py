import os
import pandas as pd

# Üst klasör
base_path = "data/images"

# Alt klasörleri gez (benign / malignant)
image_paths = []
labels = []

for label_name in ["benign", "malignant"]:
    folder_path = os.path.join(base_path, label_name)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(label_name, filename)  # sadece alt klasör + dosya adı
            image_paths.append(full_path)
            labels.append(label_name)

# DataFrame oluştur
df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})

# CSV olarak kaydet
df.to_csv("data/labels.csv", index=False)
print("✅ labels.csv başarıyla oluşturuldu.")
print(df.head())
