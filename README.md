---
title: AI Classifier Trainer
emoji: 🛠️🤖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.1.0
app_file: app.py
pinned: false
---

# 🤖 Otomatik Görüntü Sınıflandırıcı Eğitmeni 🛠️

Bu proje, sıfırdan bir görüntü sınıflandırma modelini eğitmek için gereken tüm adımları otomatikleştiren bir Gradio uygulamasıdır. Artık manuel olarak veri toplamanıza gerek yok; sadece neyi sınıflandırmak istediğinizi yazın, gerisini yapay zeka halletsin!

Bu uygulama ile istediğiniz iki kategori arasında ayrım yapabilen bir modeli dakikalar içinde oluşturup test edebilirsiniz.

## ✨ Temel Özellikler

* **Otomatik Veri Toplama:** Kullanıcının girdiği anahtar kelimelerle web'den otomatik olarak resim toplar (web scraping).
* **Sıfırdan Eğitim:** Toplanan verilerle, PyTorch kullanılarak bir Evrişimli Sinir Ağı (CNN) modeli sıfırdan eğitilir.
* **İnteraktif Test:** Eğitilen model, yeni resimlerle anında test edilebilir ve başarı oranı görülebilir.
* **Uçtan Uca Yapay Zeka Akışı:** Veri toplama, eğitim ve tahmin (inference) adımlarını tek bir arayüzde birleştirir.
* **Anlık Geri Bildirim ve Açıklamalar:** **Veri toplama ve eğitim süreçleri sırasında kullanıcıya anlık durum güncellemeleri sunar ve her adımda kullanılan teknolojiler hakkında bilgilendirici açıklamalar içerir.**

## 🚀 Kullanım Kılavuzu

Uygulama üç basit adımdan oluşur:

### **Adım 1: Veri Topla**
Uygulamanın ilk sekmesinde, sınıflandırmak istediğiniz iki kategori için arama terimleri girin (örneğin, `Isparta Gülü` ve `Lavanta Tarlası`). İndirmek istediğiniz resim sayısını seçin ve **"Veri Setini Oluştur"** butonuna tıklayın. Sistem, internetten resimleri toplayıp `train` ve `val` olarak ayıracaktır.

### **Adım 2: Modeli Eğit**
Verileriniz hazır olduğunda bu sekmeye geçin. **"Eğitimi Başlat"** butonuna tıklayarak PyTorch modelinin eğitim sürecini başlatın. Bu işlem, sunucu yoğunluğuna göre birkaç dakika sürebilir.

### **Adım 3: Tahmin Yap**
Eğitim tamamlandıktan sonra, bu son sekmeye geçerek kendi modelinizi test edebilirsiniz. Bilgisayarınızdan daha önce modelin görmediği bir resim yükleyin ve **"Tahmin Et"** butonuna basın. Modelin, yüklediğiniz resmin hangi sınıfa ait olduğunu hangi olasılıkla tahmin ettiğini göreceksiniz.

**İpucu: Her adımda neler olduğunu daha detaylı öğrenmek için sekmelerin altındaki "ℹ️ Bu Adımda Ne Oluyor?" başlığına tıklayarak ilgili açıklamaları okuyabilirsiniz.**

## 💻 Kullanılan Teknolojiler

* **Arayüz:** `Gradio`
* **Derin Öğrenme:** `PyTorch` & `torchvision`
* **Web Scraping:** `ddgs` (duckduckgo-search)
* **Veri İşleme:** `Pillow`, `NumPy`

---

Bu uygulama, kişisel bir **"AI Maratonu"**nun 27. gün projesi olarak Süleyman Toklu tarafından geliştirilmiştir.