# **🧠 Competitive Code Reasoning with LoRA (Qwen2.5-Coder)**

Bu proje, **Qwen2.5-Coder-1.5B-Instruct** temel modeli üzerinde **LoRA (Low-Rank Adaptation)** tekniği kullanılarak "Competitive Code Reasoning" (Zorlu Kodlama Problemleri) yeteneklerini geliştirmek amacıyla yapılmıştır.

Model, iki farklı veri seti (**DEEP** ve **DIVERSE**) ile ayrı ayrı eğitilmiş ve performansları karşılaştırılmıştır.

## **🚀 Modeller (Hugging Face)**

Eğitilen ve en iyi performansı veren modeller Hugging Face'e yüklenmiştir:

| Model Adı | Veri Seti | Açıklama | Link |
| :---- | :---- | :---- | :---- |
| **Qwen-Deep-LoRA** | DEEP Dataset | Karmaşık akıl yürütme gerektiren problemlerde uzman. | [Model Linki](https://huggingface.co/muzaffermut/Qwen2.5-Coder-1.5B-Deep-LoRA) |
| **Qwen-Diverse-LoRA** | DIVERSE Dataset | Çeşitli algoritma problemlerinde en iyi sonucu veren (Checkpoint-800). | [Model Linki](https://huggingface.co/muzaffermut/Qwen2.5-Coder-1.5B-Diverse-LoRA) |

## **📂 Proje Yapısı**

Lora\_Project/  
├── scripts/               \# Tüm Python kodları  
│   ├── train.py           \# LoRA eğitim kodu  
│   ├── evalconstant.py    \# Sabit seed ile model değerlendirme  
│   ├── upload.py          \# Hugging Face yükleme scripti  
│   └── download.py        \# Base model indirme  
├── outputs/               \# Eğitim logları ve model çıktıları  
├── project\_report.md      \# Detaylı proje raporu  
└── README.md              \# Bu dosya

## **🛠️ Kurulum**

Projeyi yerel ortamda çalıştırmak için:

1. **Repoyu klonlayın:**  
   git clone https://github.com/muzaffermut/Lora\_Project\_Competitive\_Code.git  
   cd Lora\_Project\_Competitive\_Code

2. **Gereksinimleri yükleyin:**  
   pip install torch transformers peft datasets bitsandbytes accelerate

## **💻 Kullanım**

### **1\. Eğitim (Training)**

Modeli eğitmek için aşağıdaki komutu kullanabilirsiniz. DEEP veya DIVERSE veri setini seçin.

python scripts/train.py \--dataset DEEP  
\# veya  
python scripts/train.py \--dataset DIVERSE

**Eğitim Parametreleri:**

* Epochs: 3  
* Batch Size: 1 (Gradient Accumulation: 16\)  
* Learning Rate: 2e-4  
* LoRA Rank: 16

### **2\. Değerlendirme (Evaluation)**

Eğitilen modeli test etmek için:

python scripts/evalconstant.py \--dataset DIVERSE \--checkpoint outputs/diverse\_model\_checkpoints/checkpoint-800

## **📊 Sonuçlar**

Yapılan testlerde:

* **DEEP Veri Seti:** Final model (846. adım) en iyi sonucu vermiştir.  
* **DIVERSE Veri Seti:** 800\. adımdaki checkpoint, final modelden daha tutarlı ve doğru kod üretmiştir (Overfitting gözlemlendiği için 800 seçildi).

Detaylı analiz için project\_report.md dosyasına bakabilirsiniz.