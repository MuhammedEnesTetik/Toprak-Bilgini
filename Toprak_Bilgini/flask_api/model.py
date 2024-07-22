import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Veriyi yükle
df = pd.read_csv(r'C:\Users\Vedat\Desktop\Toprak_Bilgini\flask_api\Toprak.csv')

# Sütun isimlerini yeniden adlandır
df = df.rename(columns={
    "Nitrogen": "Azot",
    "Phosphorus": "Fosfor",
    "Potassium": "Potasyum",
    "Temperature": "Sıcaklık",
    "Humidity": "Nem",
    "pH_Value": "pH_Değeri",
    "Rainfall": "Yağış",
    "Crop": "Ürün"
})

# İngilizce-Türkçe ürün çeviri sözlüğü
product_translation = {
    "Apple": "Elma",
    "Banana": "Muz",
    "Blackgram": "Kara fasulye",
    "ChickPea": "Nohut",
    "Coconut": "Hindistancevizi",
    "Coffee": "Kahve",
    "Cotton": "Pamuk",
    "Grapes": "Üzüm",
    "Jute": "Jüt",
    "KidneyBeans": "Kuru fasulye",
    "Lentil": "Mercimek",
    "Maize": "Mısır",
    "Mango": "Mango",
    "MothBeans": "Moth fasulyesi",
    "MungBean": "Maş fasulyesi",
    "Muskmelon": "Kavun",
    "Orange": "Portakal",
    "Papaya": "Papaya",
    "PigeonPeas": "Güvercin bezelyesi",
    "Pomegranate": "Nar",
    "Rice": "Pirinç",
    "Watermelon": "Karpuz"
}

# Ürün isimlerini Türkçeye çevir
df['Ürün'] = df['Ürün'].map(product_translation)

# Kategorik ve sayısal özellikleri belirle
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

# Etiket kodlayıcıyı uygula
label_encoder = LabelEncoder()
for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])

# Özellikleri ve hedef değişkeni ayır
y = df['Ürün']
x = df.drop(columns=['Ürün'], axis=1)

# Veriyi eğitim ve test olarak ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# Modeli eğit
rf = RandomForestClassifier()
model = rf.fit(x_train, y_train)

# Tahmin yap ve doğruluğu yazdır
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Modeli ve etiket kodlayıcıyı kaydet
joblib.dump(model, 'model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')