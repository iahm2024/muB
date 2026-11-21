# muB (Manageable Useful Bookmarks)

muB, yüzlerce GitHub reposu arasında kaybolmayı önlemek için geliştirilmiş, %100 yerel (local) çalışan, yapay zeka destekli bir yer imi yöneticisidir. 

Standart yer imi yöneticilerinden farklı olarak muB, kaydettiğiniz repoların README dosyalarını ve açıklamalarını vektör uzayına taşır. Bu sayede repoları sadece isimleriyle değil, "anlamsal" (semantic) olarak arayabilir, Qwen LLM ile otomatik kategorize edebilir ve kendi veritabanınız üzerinde RAG (Retrieval-Augmented Generation) çalıştırabilirsiniz.

## 🚀 Özellikler

- **Yerel ve Gizli:** Vektör veritabanı (SQLite-vec) ve yapay zeka modelleri (Llama.cpp) tamamen kendi bilgisayarınızda çalışır. Verileriniz dışarı çıkmaz.
- **Hibrit Arama (Score Fusion):** Arama sonuçları; Kosinüs Benzerliği (Cosine Similarity), kelime frekansı ve repoların yıldız sayıları (logaritmik) harmanlanarak en alakalıdan en aza doğru sıralanır.
- **Akıllı Kategorizasyon:** Arayüzden eklenen her yeni GitHub reposu, Qwen LLM tarafından incelenir ve içeriğine göre tek kelimelik (Frontend, Database, AI vb.) bir etiketle otomatik sınıflandırılır. Hugging Face modelleri ise LLM'e verilmeden kendi görev etiketiyle (Text-generation, Sentence-similarity vb.) kategorilenir; etiket yoksa "Genel" olur.
- **Hugging Face Model Desteği:** GitHub repolarının yanında Hugging Face model linkleri de eklenebilir; model kartı (README) vektörlenir, beğeni ve indirme sayıları senkronize edilir.
- **Gelişmiş Arayüz:** Streamlit ile geliştirilen arayüzde; GitHub ve Hugging Face yer imleri için ayrı sayfalar, kaynağa göre süzülebilen yapay zeka araması, tembel yükleme (lazy loading) ile anında açılan README okuyucu, yazara veya kategoriye göre gruplama, toplu etiket düzenleme ve GitHub API üzerinden tek tıkla yıldız/fork senkronizasyonu bulunur.

## 🛠️ Kurulum

### 1. Projeyi Klonlayın ve Bağımlılıkları Kurun
```bash
git clone https://github.com/iahm2024/muB.git
cd muB
pip install -r requirements.txt
```

### 2. Modelleri İndirin
Sistemin çalışması için iki adet `.gguf` formatında modele ihtiyacınız var. Modelleri indirip `data/models/` klasörüne (veya istediğiniz bir klasöre) yerleştirin:
- **Embedding Modeli:** [nomic-embed-text-v1.5.Q8_0.gguf](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) (Metinleri vektörleştirmek için)
- **LLM Modeli:** [qwen2.5-coder-3b-instruct-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF) (Kategorizasyon ve sohbet için)

### 3. Çevresel Değişkenleri (.env) Ayarlayın
Proje ana dizininde bir `.env` dosyası oluşturun ve aşağıdaki bilgileri kendi sisteminize göre doldurun:
```text
GITHUB_TOKEN=ghp_sizin_github_tokeniniz
HF_TOKEN=hf_sizin_tokeniniz
EMBEDDING_MODEL_PATH=data/models/nomic-embed-text-v1.5.Q8_0.gguf
LLM_MODEL_PATH=data/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
```
*(Model yolları, komutları çalıştırdığınız proje ana dizinine göredir.)*
*(Not: GitHub Token, GitHub API'nin hız sınırlarına takılmadan repo verilerini çekebilmek için zorunludur.)*
*(Not: HF_TOKEN isteğe bağlıdır; yalnız kısıtlı (gated) veya özel modeller için gereklidir.)*

## 💻 Kullanım

Tüm kurulumları tamamladıktan sonra arayüzü başlatmak için proje ana dizininde şu komutu girin:
```bash
streamlit run app/app.py
```
* Tarayıcınızda açılan sekmeden sisteme tek tek repo veya Hugging Face model linki yapıştırabilir, bir yazarın/şirketin tüm repolarını topluca çekebilir veya yapay zekaya sorular sorabilirsiniz.
* Geçmişten kalan `.html` formatındaki tarayıcı yer imlerinizi (bookmarks) içe aktarmak isterseniz (GitHub linklerinin yanında Hugging Face model linkleri de alınır), `data/bookmarks.html` yoluna dosyanızı koyup proje ana dizininde `python app/ingest.py` betiğini çalıştırabilirsiniz.

## 🏗️ Kullanılan Teknolojiler

- **Arayüz:** Streamlit
- **Vektör Veritabanı:** SQLite + sqlite-vec
- **Yapay Zeka:** Llama.cpp (Python bindings)
- **Veri Kaynağı:** GitHub REST API, Hugging Face Hub API

## 📄 Lisans
Bu proje MIT lisansı altında açık kaynak olarak paylaşılmıştır. Dilediğiniz gibi kullanabilir ve geliştirebilirsiniz.