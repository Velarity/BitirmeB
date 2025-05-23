import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Etiket dosyasını oku
df = pd.read_csv("data/labels.csv")

# Tam görsel yolunu oluştur
df['full_path'] = df['image_path'].apply(lambda x: os.path.join("data/images", x))

# Etiketleri sayısal hale getir (benign: 0, malignant: 1)
df['label_encoded'] = df['label'].map({'benign': 0, 'malignant': 1})

# Boş veya geçersiz satır varsa temizle
df = df.dropna()
df = df[df['full_path'].apply(lambda x: os.path.exists(x))]

# Eğitim ve test setlerine ayır
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label_encoded'],
    random_state=42
)

# Kaydet
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# Bilgi mesajı
print(f"✅ Eğitim veri sayısı: {len(train_df)}")
print(f"✅ Test veri sayısı: {len(test_df)}")
print("✅ Veriler başarıyla hazırlandı ve kaydedildi.")
