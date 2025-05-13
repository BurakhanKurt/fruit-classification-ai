import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from train import FruitClassifier
import os

def load_model(model_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FruitClassifier(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(image_path, model, class_names, device='cuda'):
    # Görüntüyü yükle ve dönüştür
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Tahmin yap
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

def main():
    # Model ve sınıf isimlerini yükle
    model_path = 'fruit_classifier.pth'
    class_names = sorted(os.listdir('data/Training'))  # Sınıf isimlerini al
    
    model = load_model(model_path, len(class_names))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test klasöründeki tüm görüntüleri tahmin et
    test_dir = 'data/Test'
    results = []
    
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            predicted_class, confidence = predict_image(image_path, model, class_names, device)
            
            results.append({
                'image': image_name,
                'true_class': class_name,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'correct': class_name == predicted_class
            })
    
    # Sonuçları yazdır
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = 100. * correct / total
    
    print(f'\nToplam Test Görüntüsü: {total}')
    print(f'Doğru Tahmin: {correct}')
    print(f'Doğruluk: {accuracy:.2f}%')
    
    # Yanlış tahminleri göster
    print('\nYanlış Tahminler:')
    for r in results:
        if not r['correct']:
            print(f"Görüntü: {r['image']}")
            print(f"Gerçek Sınıf: {r['true_class']}")
            print(f"Tahmin Edilen: {r['predicted_class']} (Güven: {r['confidence']:.2f})")
            print('---')

if __name__ == '__main__':
    main() 