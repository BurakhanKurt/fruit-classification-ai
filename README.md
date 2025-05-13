# 🍎 Yapay Zeka Destekli Meyve Sınıflandırma Uygulaması

Bu proje, meyve görüntülerini sınıflandırmak için derin öğrenme kullanan bir yapay zeka uygulamasıdır. Fruits 360 veri seti kullanılarak eğitilmiş bir model ile 130'dan fazla farklı meyve sınıfını tanıyabilir.

## 🚀 Özellikler

- Kullanıcı dostu web arayüzü
- Gerçek zamanlı meyve sınıflandırma
- Yüksek doğruluk oranı
- Detaylı güven skorları
- Desteklenen formatlar: JPG, JPEG, PNG

## 📋 Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelere ihtiyacınız var:

```bash
torch>=2.7.0
torchvision>=0.22.0
numpy>=1.26.0
Pillow>=10.2.0
streamlit>=1.32.0
matplotlib>=3.8.0
tqdm>=4.66.0
```

## 🛠️ Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/kullaniciadi/meyve-siniflandirma.git
cd meyve-siniflandirma
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Fruits 360 veri setini indirin ve `data` klasörüne yerleştirin:
   - [Fruits 360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)'i indirin
   - `Training` ve `Test` klasörlerini `data` dizinine kopyalayın

## 🚀 Kullanım

1. Modeli eğitin:
```bash
python src/train.py
```

2. Web arayüzünü başlatın:
```bash
streamlit run src/app.py
```

3. Tarayıcınızda açılan arayüzde:
   - "Bir meyve fotoğrafı yükleyin" butonuna tıklayın
   - Bir meyve fotoğrafı seçin
   - "Tahmin Et" butonuna tıklayın
   - Sonuçları görüntüleyin

## 🏗️ Proje Yapısı

```
meyve-siniflandirma/
├── data/
│   ├── Training/
│   └── Test/
├── src/
│   ├── train.py
│   └── app.py
├── requirements.txt
└── README.md
```

## 🧠 Model Mimarisi

Proje, özel olarak tasarlanmış bir CNN (Convolutional Neural Network) mimarisi kullanır:

- 3 konvolüsyon bloğu
- Batch Normalization
- Dropout katmanları
- ReLU aktivasyon fonksiyonları
- AdamW optimizer
- Learning Rate Scheduling

## 📊 Performans Metrikleri

Model eğitimi sırasında aşağıdaki metrikler takip edilir:
- Eğitim kaybı (Training Loss)
- Doğrulama kaybı (Validation Loss)
- Eğitim doğruluğu (Training Accuracy)
- Doğrulama doğruluğu (Validation Accuracy)

## 🎯 Teknik Detaylar

### Veri Ön İşleme
- Görüntü boyutlandırma (64x64)
- Normalizasyon
- Veri artırma (Data Augmentation):
  - Yatay çevirme
  - Küçük rotasyonlar
  - Renk değişimleri

### Model Eğitimi
- Epoch sayısı: 50
- Batch size: 32
- Öğrenme oranı: 0.001 (AdamW)
- Weight decay: 0.01
- Learning Rate Scheduling: ReduceLROnPlateau

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👥 Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: X'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Bir Pull Request oluşturun

## 📞 İletişim

Sorularınız veya önerileriniz için [GitHub Issues](https://github.com/kullaniciadi/meyve-siniflandirma/issues) üzerinden iletişime geçebilirsiniz. 