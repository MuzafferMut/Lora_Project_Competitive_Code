import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Kullanılacak Model
model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print(f"--- MODEL BAŞLATILIYOR ---\nModel: {model_id}")

# 1. Modeli Yükleme (Disk -> GPU)
try:
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # RTX 5060 performansı için
        device_map="auto"           # Otomatik olarak GPU'ya atar
    )
    
    load_time = time.time() - start_time
    print(f"✅ Model GPU'ya yüklendi ({load_time:.2f} saniye sürdü).")
    
except Exception as e:
    print(f"❌ HATA: Model yüklenemedi. Önce download.py çalıştırdın mı?\nHata: {e}")
    exit()

# 2. Test Sorusu (Hocanın istediği inference testi)
prompt = "Write a Python function to calculate the Fibonacci sequence up to n terms."

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": prompt}
]

# 3. Hazırlık ve Üretim
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

print("\n🤖 Model Düşünüyor (Inference)...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# 4. Çıktıyı Temizleme
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("-" * 40)
print(f"SORU: {prompt}")
print("-" * 40)
print(response)
print("-" * 40)