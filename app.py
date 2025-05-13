import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# Sayfa yapılandırması
st.set_page_config(
    page_title="Görüntü Sınıflandırıcı",
    page_icon="🖼️",
    layout="centered"
)

# Başlık ve açıklama
st.title("Yapay Zeka Destekli Görüntü Sınıflandırıcı")
st.write("Bu uygulama, yüklediğiniz görüntüleri 10 farklı kategoride sınıflandırır.")

# Sınıf isimleri
classes = ('uçak', 'araba', 'kuş', 'kedi', 'geyik',
           'köpek', 'kurbağa', 'at', 'gemi', 'kamyon')

# Model yükleme fonksiyonu
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('cifar10_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Görüntü ön işleme fonksiyonu
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

# Tahmin fonksiyonu
def predict(image):
    model = load_model()
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    return predicted_class, confidence

# Ana uygulama
def main():
    uploaded_file = st.file_uploader("Bir görüntü seçin", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Görüntüyü göster
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Görüntü", use_column_width=True)
        
        # Tahmin butonu
        if st.button("Sınıflandır"):
            with st.spinner("Sınıflandırılıyor..."):
                predicted_class, confidence = predict(image)
                
                # Sonuçları göster
                st.success(f"Tahmin: {classes[predicted_class]}")
                st.info(f"Güven: {confidence*100:.2f}%")
                
                # Güven skorlarını göster
                st.write("Tüm sınıflar için güven skorları:")
                model = load_model()
                input_tensor = preprocess_image(image)
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
                    st.progress(float(prob))
                    st.write(f"{class_name}: {prob*100:.2f}%")

if __name__ == "__main__":
    main() 