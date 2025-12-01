import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random

# --- AYARLAR ---
SYSTEM_PROMPT = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."

DATASETS = {
    "DEEP": "Naholav/CodeGen-Deep-5K",
    "DIVERSE": "Naholav/CodeGen-Diverse-5K"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["DEEP", "DIVERSE"], help="Hangi datasetin test verisi?")
    parser.add_argument("--checkpoint", type=str, required=True, help="Test edilecek model klasörünün yolu")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 🛑 KRİTİK EKLEME: SABİT SEED
    # Bunu eklediğimiz için artık her seferinde AYNI soruları seçecek.
    random.seed(42)
    
    print(f"\n🔍 MODEL DEĞERLENDİRİLİYOR (Adil Test)...")
    print(f"Dataset: {args.dataset}")
    print(f"Model Yolu: {args.checkpoint}")
    
    # 1. Modeli Yükle
    try:
        print("-> Model ve Tokenizer yükleniyor...")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    except Exception as e:
        print(f"❌ HATA: Model yüklenemedi. Klasör yolunu doğru yazdın mı?\n{e}")
        return

    # 2. Test Verisini İndir
    print("-> Test verisi hazırlanıyor...")
    full_dataset = load_dataset(DATASETS[args.dataset], split="train")
    
    # Eğitimdekiyle aynı ayrımı yapıyoruz
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = split_dataset["test"]
    
    print(f"-> Toplam {len(test_dataset)} test sorusu var. (Seed: 42 ile sabit 3 soru seçiliyor)")
    
    # Rastgele ama SABİT 3 indeks seç
    indices = random.sample(range(len(test_dataset)), 3)
    
    for i in indices:
        example = test_dataset[i]
        input_text = example["input"]
        ground_truth = example["solution"]
        
        # Prompt Hazırla
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print(f"\n{'='*40}")
        print(f"SORU (Özet): {input_text[:500]}...")
        print(f"{'-'*40}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024,
                temperature=0.2, 
                top_p=0.9,
                do_sample=True
            )
        
        generated_code = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"🤖 MODELİN CEVABI:\n{generated_code}")
        print(f"\n✅ GERÇEK CEVAP (Referans):\n{ground_truth}")
        print(f"{'='*40}\n")

    print("Test Tamamlandı.")

if __name__ == "__main__":
    main()