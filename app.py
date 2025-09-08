import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import gradio as gr
from PIL import Image
import numpy as np
from pathlib import Path
import requests
from io import BytesIO
import shutil
import time
import re
from ddgs import DDGS

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    return text

def download_images(search_term, folder, max_images=40):
    path = Path(folder)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, parents=True)
    
    downloaded_count = 0
    with DDGS() as ddgs:
        ddgs_images_gen = ddgs.images(
            query=search_term,
            region='tr-tr',
            max_results=int(max_images * 1.5)
        )
        
        for r in ddgs_images_gen:
            if downloaded_count >= max_images:
                break
            try:
                response = requests.get(r['image'], timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img.save(path / f"{downloaded_count + 1}.jpg")
                downloaded_count += 1
            except Exception as e:
                pass 
            
    return f"'{folder}' klasörüne {downloaded_count} adet resim indirildi."

def setup_dataset(class1_query, class2_query, num_images_per_class, split_ratio=0.8):
    base_dir = Path('dataset')
    if base_dir.exists():
        shutil.rmtree(base_dir)

    yield "Veri seti klasörleri temizleniyor ve hazırlanıyor..."

    class1_name = slugify(class1_query)
    class2_name = slugify(class2_query)
    
    temp_class1_dir = base_dir / "temp" / class1_name
    temp_class2_dir = base_dir / "temp" / class2_name
    
    yield f"'{class1_query}' için indirme başlıyor..."
    msg1 = download_images(class1_query, temp_class1_dir, num_images_per_class)
    yield f"'{class1_query}' için indirme tamamlandı. ({msg1})"
    
    time.sleep(2) 
    
    yield f"'{class2_query}' için indirme başlıyor..."
    msg2 = download_images(class2_query, temp_class2_dir, num_images_per_class)
    yield f"'{class2_query}' için indirme tamamlandı. ({msg2})"

    yield "İndirilen resimler 'train' ve 'val' olarak ayrılıyor..."
    for phase in ['train', 'val']:
        (base_dir / phase / class1_name).mkdir(exist_ok=True, parents=True)
        (base_dir / phase / class2_name).mkdir(exist_ok=True, parents=True)
        
    def split_files(source_dir, train_dir, val_dir):
        files = list(source_dir.glob("*.jpg"))
        split_point = int(len(files) * split_ratio)
        train_files, val_files = files[:split_point], files[split_point:]
        for f in train_files: shutil.copy(f, train_dir / f.name)
        for f in val_files: shutil.copy(f, val_dir / f.name)
            
    split_files(temp_class1_dir, base_dir / 'train' / class1_name, base_dir / 'val' / class1_name)
    split_files(temp_class2_dir, base_dir / 'train' / class2_name, base_dir / 'val' / class2_name)
    
    shutil.rmtree(base_dir / "temp")
    
    yield f"🎉 Veri seti başarıyla oluşturuldu! Sınıflar: '{class1_name}', '{class2_name}'"

def train_model(data_dir, num_epochs=10):
    yield "Eğitim başlıyor... Veriler yükleniyor..."
    
    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(128), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize(150), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    with open("class_names.txt", "w") as f: f.write(",".join(class_names))
    
    yield f"Veriler yüklendi. Sınıflar: {class_names}. Cihaz hazırlanıyor..."
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    yield f"Model eğitime hazır. Cihaz: {device}. Toplam Epoch: {num_epochs}"

    for epoch in range(num_epochs):
        epoch_info = f'Epoch {epoch+1}/{num_epochs}\n' + ('-' * 20)
        
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            epoch_info += f'\n{phase.capitalize()} | Kayıp: {epoch_loss:.4f} - Başarı: {epoch_acc:.4f}'
            yield epoch_info

    torch.save(model.state_dict(), 'model.pth')
    
    global TRAINED_MODEL, CLASS_NAMES
    TRAINED_MODEL = model
    CLASS_NAMES = class_names

    yield f"✅ Eğitim tamamlandı! Model 'model.pth' olarak kaydedildi."

def start_training_wrapper():
    if not os.path.exists('dataset/train'):
        return "Hata: Önce veri setini oluşturmalısınız."
    yield from train_model('dataset')

TRAINED_MODEL, CLASS_NAMES = None, None
MODEL_PATH, CLASS_NAMES_PATH = "model.pth", "class_names.txt"

def predict(image):
    if TRAINED_MODEL is None or CLASS_NAMES is None: return "Model henüz eğitilmedi veya yüklenemedi."
    transform = transforms.Compose([transforms.Resize(150), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = transform(image).unsqueeze(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    with torch.no_grad():
        outputs = TRAINED_MODEL(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    return confidences

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Otomatik Görüntü Sınıflandırıcı Eğitmeni 🛠️")
    gr.Markdown("Bu uygulama, internetten belirttiğiniz iki sınıf için resim toplar, bir model eğitir ve tahmin yapmanızı sağlar.")

    with gr.Tab("Adım 1: Veri Topla"):
        class1_input = gr.Textbox(label="Sınıf 1 Arama Terimi", placeholder="Örn: Isparta Gülü")
        class2_input = gr.Textbox(label="Sınıf 2 Arama Terimi", placeholder="Örn: Lavanta Tarlası")
        num_images_slider = gr.Slider(minimum=20, maximum=100, value=40, step=5, label="Her Sınıf İçin İndirilecek Resim Sayısı")
        data_button = gr.Button("Veri Setini Oluştur")
        data_output = gr.Textbox(label="Veri Toplama Durumu", interactive=False, lines=5)
        data_button.click(setup_dataset, inputs=[class1_input, class2_input, num_images_slider], outputs=data_output)
        
        with gr.Accordion("ℹ️ Bu Adımda Ne Oluyor?", open=False):
            gr.Markdown("""
            Bu adımda, girdiğiniz arama terimleri kullanılarak internetten otomatik olarak resimler indirilir.
            - **Kütüphane:** `ddgs` (DuckDuckGo Search)
            - **İşlem:** Belirtilen sayıda resim bulunur, indirilir ve `dataset` klasörü altında `train` (eğitim) ve `val` (doğrulama) olarak ikiye ayrılır.
            - **`time.sleep(2)`:** İki arama arasında eklenen bu küçük bekleme, arama motoru tarafından 'rate limit'e (hız sınırına) takılmamızı engeller.
            """)

    with gr.Tab("Adım 2: Modeli Eğit"):
        train_button = gr.Button("Eğitimi Başlat")
        train_output = gr.Textbox(label="Eğitim Durumu", interactive=False, lines=10)
        train_button.click(start_training_wrapper, outputs=train_output)

        with gr.Accordion("ℹ️ Bu Adımda Ne Oluyor?", open=False):
            gr.Markdown("""
            Bu adımda, topladığımız resimleri kullanarak bir **Evrişimli Sinir Ağı (Convolutional Neural Network - CNN)** modelini sıfırdan eğitiyoruz.
            - **Kütüphane:** `PyTorch`
            - **Epoch:** Veri setinin tamamının model tarafından bir kez işlenmesi demektir.
            - **Loss (Kayıp):** Modelin tahminlerinin ne kadar 'yanlış' olduğunu ölçen bir değerdir. **Düşük olması iyidir.**
            - **Accuracy (Başarı):** Modelin tahminlerinin ne kadarının 'doğru' olduğunu gösteren bir orandır. **Yüksek olması iyidir.**
            """)

    with gr.Tab("Adım 3: Tahmin Yap"):
        image_input = gr.Image(type="numpy", label="Bir Resim Yükleyin")
        label_output = gr.Label(num_top_classes=2)
        predict_button = gr.Button("Tahmin Et")
        predict_button.click(predict, inputs=image_input, outputs=label_output)

        with gr.Accordion("ℹ️ Bu Adımda Ne Oluyor?", open=False):
            gr.Markdown("""
            Bu adımda, az önce eğittiğimiz model ile daha önce hiç görmediği bir resmi sınıflandırıyoruz. Bu işleme **çıkarım (inference)** denir.
            - **İşlem:** Yüklenen resim, modelin anlayacağı formata dönüştürülür ve eğittiğimiz sinir ağından geçirilir.
            - **Sonuç:** Model, resmin her bir sınıfa ait olma olasılığını bir yüzde olarak hesaplar (`Softmax`). En yüksek olasılığa sahip sınıf, modelin tahmini olarak kabul edilir.
            """)

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        TRAINED_MODEL = SimpleCNN().to(device)
        TRAINED_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        TRAINED_MODEL.eval()
        with open(CLASS_NAMES_PATH, "r") as f: CLASS_NAMES = f.read().strip().split(',')
        print("Önceden eğitilmiş model ve sınıflar başarıyla yüklendi.")
except Exception as e:
    TRAINED_MODEL, CLASS_NAMES = None, None
    print(f"Önceden eğitilmiş model yüklenirken hata oluştu: {e}")

if __name__ == "__main__":
    demo.launch()