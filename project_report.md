# **LoRA Fine-Tuning Projesi: Competitive Code Reasoning**

Öğrenci Adı/No: Muzaffer MUT 2021556048  
Tarih: 02.12.2025  
Bu proje, Hugging Face platformundan indirilen Qwen2.5-Coder-1.5B-Instruct temel modeli üzerinde, LoRA (Low-Rank Adaptation) tekniği kullanılarak "Competitive Code Reasoning" yeteneklerini geliştirmek amacıyla gerçekleştirilmiştir.

## **I. Temel Model ve Veri Setleri**

| Bileşen | Detay |
| :---- | :---- |
| **Temel Model** | Qwen/Qwen2.5-Coder-1.5B-Instruct (1.5 Milyar Parametre) |
| **Fine-Tuning Tekniği** | LoRA (Low-Rank Adaptation) |
| **Donanım** | NVIDIA GeForce RTX 5060 (8 GB VRAM) |
| **Yazılım Ortamı** | PyTorch, Hugging Face Transformers, PEFT, CUDA 12.x |

## **II. Eğitim Detayları ve Hyperparameter'lar (Görev 5.3)**

İki ayrı eğitim, Base Model'den bağımsız olarak başlatılmıştır. VRAM kısıtlaması nedeniyle özel optimizasyonlar uygulanmıştır.

### **Optimizasyonlar**

* **Tensor Tipi:** bfloat16 (RTX 50 serisinin hızını kullanmak için)  
* **Attention Mekanizması:** PyTorch SDPA (Flash Attention 2 kütüphane uyumsuzluğu nedeniyle)  
* **VRAM Tasarrufu:** gradient\_checkpointing\_enable() aktif edilmiştir.  
* **Batch Size Hilesi:** per\_device\_train\_batch\_size=1 ve gradient\_accumulation\_steps=16 kullanılarak **Effective Batch Size 16** sağlanmıştır.

### **LoRA Parametreleri**

| Parametre | Değer | Açıklama |
| :---- | :---- | :---- |
| lora\_r (Rank) | 16 | LoRA adaptör matris boyutu. |
| lora\_alpha | 32 | r \* 2 kuralına uygunluk (PDF önerisi). |
| lora\_dropout | 0.05 | Aşırı öğrenmeyi engellemek için. |
| target\_modules | Tüm lineer katmanlar (q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, down\_proj) |  |

### **Training Argümanları**

| Parametre | Değer | PDF Gereksinimi |
| :---- | :---- | :---- |
| num\_train\_epochs | 3 | Zorunlu 3 |
| learning\_rate (lr) | 2e-4 | Standart LoRA değeri |
| weight\_decay | 0.01 |  |
| logging\_steps | 20 | 20-40 aralığına uygun |
| eval\_steps | 100 | 100-120 aralığına uygun |
| optim | adamw\_torch |  |
| lr\_scheduler\_type | cosine |  |
| warmup\_ratio | 0.03 |  |
| **Zorunlu Prompt** | "You are an expert Python programmer..." | Zorunlu System Prompt |

## **III. Sonuçlar ve Checkpoint Seçimi (Görev 4\)**

Her iki eğitim için de, son checkpoint'ler ve sondan önceki checkpoint'ler (checkpoint-800 ve checkpoint-700) **aynı test sorularıyla** karşılaştırılmıştır.

### **1\. DEEP Dataset**

**Seçilen Checkpoint:** **deep\_final\_model**

**Analiz:** Karşılaştırmalı testlerde, DEEP Final Modeli (846. adım) ve diğer checkpoint'ler arasında büyük bir zeka farkı gözlemlenmemiştir. Bu nedenle en uzun eğitim gören (final\_model) versiyon yüklenmiştir.

### **2\. DIVERSE Dataset**

**Seçilen Checkpoint:** **checkpoint-800**

**Analiz:** DIVERSE Final Modeli, zorlu "Kale (Rook) Sorunu" ve "Garson Para Üstü Sorunu" gibi problemlerde, 800\. adımdaki haline göre daha karmaşık ve mantıksal olarak daha hatalı çözümler üretmiştir. Bu, modelin son adımlarda aşırı ezberlemeye başladığını (overfitting) göstermiştir. En iyi performansı veren 800\. adım yüklenmiştir.

## **IV. Teslim Edilen Modeller (Görev 5.2)**

Bu modeller, seçilen en iyi checkpoint'lerin LoRA adaptörleri ile birleştirilmiş halidir (veya LoRA adaptörleri ayrı yüklenmiştir).

| Dataset | Yüklenen Checkpoint | Hugging Face Repo Linki |
| :---- | :---- | :---- |
| **DEEP** | deep\_final\_model | https://huggingface.co/muzaffermut/Qwen2.5-Coder-1.5B-Deep-LoRA |
| **DIVERSE** | checkpoint-800 | https://huggingface.co/muzaffermut/Qwen2.5-Coder-1.5B-Diverse-LoRA |

## 

## 

## **V. Training Logları (Görev 5.1)**

Loglar, [https://github.com/MuzafferMut/Lora\_Project\_Competitive\_Code](https://github.com/MuzafferMut/Lora_Project_Competitive_Code) linkindeki outputs klasöründe bulunmaktadır. Log formatı, hocanın isteği üzerine her 20 adımda bir train\_loss ve her 100 adımda bir eval\_loss içerecek şekilde ayarlanmıştır.

**GitHub Repository Linki:** [https://github.com/MuzafferMut/Lora\_Project\_Competitive\_Code](https://github.com/MuzafferMut/Lora_Project_Competitive_Code) 