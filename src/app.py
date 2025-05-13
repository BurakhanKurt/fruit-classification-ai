import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from train import FruitClassifier

# Sayfa yapılandırması
st.set_page_config(
    page_title="Meyve Sınıflandırma",
    page_icon="🍎",
    layout="centered"
)

# Başlık ve açıklama
st.title("🍎 Meyve Sınıflandırma Uygulaması")
st.write("Bu uygulama, yüklediğiniz meyve fotoğraflarını sınıflandırır.")

# Model yükleme fonksiyonu
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'fruit_classifier.pth'
    class_names = sorted(os.listdir('data/Training'))
    model = FruitClassifier(len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, class_names

# Görüntü tahmin fonksiyonu
def predict_image(image, model, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

# Model yükleme
try:
    model, class_names = load_model()
    st.success("Model başarıyla yüklendi!")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {str(e)}")
    st.stop()

# Dosya yükleme
uploaded_file = st.file_uploader("Bir meyve fotoğrafı yükleyin", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Görüntüyü göster
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Yüklenen Görüntü", width=300)
    
    # Tahmin yap
    if st.button("Tahmin Et"):
        with st.spinner("Tahmin yapılıyor..."):
            predicted_class, confidence = predict_image(image, model, class_names)
            
            # Sonuçları göster
            st.success(f"Tahmin: {predicted_class}")
            st.info(f"Güven: {confidence:.2%}")
            
            # Güven çubuğu
            st.progress(confidence)

# Bilgi kutusu
with st.expander("Hakkında"):
    st.write("""
    Bu uygulama, Fruits 360 veri seti üzerinde eğitilmiş bir derin öğrenme modeli kullanarak 
    meyve fotoğraflarını sınıflandırır. Model, 130'dan fazla farklı meyve sınıfını tanıyabilir.
    
    ### Nasıl Kullanılır?
    1. "Bir meyve fotoğrafı yükleyin" butonuna tıklayın
    2. Bir meyve fotoğrafı seçin
    3. "Tahmin Et" butonuna tıklayın
    4. Sonuçları görüntüleyin
    
    ### Desteklenen Formatlar
    - JPG
    - JPEG
    - PNG
    """) 