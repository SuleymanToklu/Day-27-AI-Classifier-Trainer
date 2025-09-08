---
title: AI Music Studio
emoji: 🎶
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
---

Gün 26: AI Music Studio - Demucs ile Enstrüman Ayırma 🎶

Bu proje, bir ses dosyasını alıp demucs kütüphanesini kullanarak onu temel enstrüman katmanlarına (vokal, davul, bas ve diğerleri) ayıran basit ve güçlü bir Gradio web uygulamasıdır.

Uygulama, transformers pipeline'ı yerine doğrudan demucs komut satırı aracını çağırarak daha stabil ve güvenilir bir ayırma işlemi sunar.
🚀 Canlı Demo

Uygulamayı denemek için aşağıdaki Hugging Face Spaces linkini ziyaret edebilirsiniz:

https://huggingface.co/spaces/tiheli/Day-26-AI-Music-Studio
✨ Özellikler

    🎼 Stabil Enstrüman Ayırma: Yüklediğiniz bir ses dosyasını vokal, davul, bas ve diğer enstrümanlar olmak üzere dört ana katmana ayırır.

    Doğrudan Demucs Entegrasyonu: Arka planda subprocess kullanarak demucs kütüphanesini çalıştırır ve en iyi sonuçları hedefler.

    Basit Arayüz: Tek yapmanız gereken ses dosyanızı yüklemek ve sonuçları dinlemek!

🛠️ Kullanılan Teknolojiler

    Framework: Gradio

    Ayırma Kütüphanesi: Demucs

    Ana Kütüphaneler: subprocess, tempfile (Geçici dosya yönetimi için)

💻 Yerel Ortamda Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

    Repoyu Klonlayın:

    git clone [https://github.com/SuleymanToklu/Day26-AI-Music-Studio.git](https://github.com/SuleymanToklu/Day26-AI-Music-Studio.git)
    cd Day26-AI-Music-Studio

    Gerekli Paketleri Yükleyin:
    demucs kütüphanesinin kendisi ana bağımlılıktır.

    pip install -r requirements.txt

    Not: requirements.txt dosyanızın gradio ve demucs paketlerini içerdiğinden emin olun. Demucs'un kurulumu için resmi GitHub sayfasını ziyaret edebilirsiniz.

    Uygulamayı Başlatın:

    python app.py

    Tarayıcıda Açın:
    Terminalde belirtilen yerel adresi (genellikle http://127.0.0.1:7860) tarayıcınızda açın.