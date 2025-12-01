import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# İndirilecek Modelin Kimliği
model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print(f"--- İNDİRME BAŞLIYOR ---\nModel: {model_id}")
print("Dosyalar Hugging Face cache klasörüne indiriliyor...")

try:
    # 1. Tokenizer'ı indir
    print("-> Tokenizer indiriliyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. Modeli indir (Sadece indirir, belleğe tam yükleme yapmaz)
    print("-> Model ağırlıkları indiriliyor...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True # RAM'i şişirmeden indirmek için
    )
    
    print("\n✅ BAŞARILI: Model ve Tokenizer başarıyla indirildi ve cache'lendi.")
    print(f"Konum: {os.path.expanduser('~/.cache/huggingface/hub')}")

except Exception as e:
    print(f"\n❌ HATA: İndirme sırasında bir sorun oluştu:\n{e}")