import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# --- 1. AYARLAR VE PARAMETRELER ---
# Hocanın PDF'teki zorunlu system prompt'u
SYSTEM_PROMPT = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."

# Dataset İsimleri
DATASETS = {
    "DEEP": "Naholav/CodeGen-Deep-5K",
    "DIVERSE": "Naholav/CodeGen-Diverse-5K"
}

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Training Script")
    parser.add_argument("--dataset", type=str, required=True, choices=["DEEP", "DIVERSE"], help="Hangi dataset ile eğitim yapılacak?")
    return parser.parse_args()

def format_chat_template(row, tokenizer):
    """
    Veriyi Qwen modelinin anlayacağı chat formatına çevirir.
    Sadece 'input' (soru) ve 'solution' (temiz kod) kullanılır.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["input"]},
        {"role": "assistant", "content": row["solution"]} # Sadece SOLUTION kullanıyoruz
    ]
    
    # Tokenizer ile chat formatını uygula ve ID'lere çevir
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Modeli girdiyi tokenlara çevir
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=1024, # PDF'te kod için önerilen uzunluk
        padding="max_length",
    )
    
    # Labels oluştur (Modelin neyi tahmin edeceğini belirler)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def main():
    args = parse_args()
    selected_dataset = DATASETS[args.dataset]
    output_dir = f"outputs/{args.dataset.lower()}_model_checkpoints"
    
    print(f"\n🚀 EĞİTİM BAŞLIYOR: {args.dataset} Dataset")
    print(f"Model: Qwen2.5-Coder-1.5B-Instruct")
    print(f"Çıktı Klasörü: {output_dir}")
    
    # --- 2. MODEL VE TOKENIZER YÜKLEME ---
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 

    # Model Yükleme (VRAM Tasarrufu için ayarlar)
    print("-> Model yükleniyor (Bfloat16)...")
    
    # DÜZELTME: Flash Attention zorunluluğunu kaldırdık.
    # PyTorch otomatik optimize edecek.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        use_cache=False, 
    )
    
    # Gradient Checkpointing (VRAM tasarrufu için kritik)
    model.gradient_checkpointing_enable()
    
    # --- 3. LoRA AYARLARI ---
    peft_config = LoraConfig(
        r=16,               # Rank
        lora_alpha=32,      # Alpha
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # --- 4. DATASET HAZIRLIĞI ---
    print(f"-> Dataset indiriliyor: {selected_dataset}")
    dataset = load_dataset(selected_dataset, split="train")
    
    # Eğitim ve Test olarak böl (%10 validation)
    dataset = dataset.train_test_split(test_size=0.1)
    
    print("-> Veriler formatlanıyor...")
    train_dataset = dataset["train"].map(
        lambda x: format_chat_template(x, tokenizer),
        batched=False,
        remove_columns=dataset["train"].column_names
    )
    eval_dataset = dataset["test"].map(
        lambda x: format_chat_template(x, tokenizer),
        batched=False,
        remove_columns=dataset["test"].column_names
    )

    # --- 5. EĞİTİM ARGÜMANLARI ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,              
        per_device_train_batch_size=1,   # VRAM için mecburi 1
        gradient_accumulation_steps=16,  # 1 * 16 = 16 Effective Batch Size
        learning_rate=2e-4,              
        weight_decay=0.01,
        logging_steps=20,                
        eval_strategy="steps",           
        eval_steps=100,                  
        save_strategy="steps",
        save_steps=100,                  
        save_total_limit=3,              
        fp16=False,
        bf16=True,                       # RTX 5060 için True
        report_to="none",                
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )

    # --- 6. TRAINER ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    
    print("-> Eğitim Başlıyor! (Bu işlem saatler sürebilir)")
    trainer.train()
    
    # Final Modeli Kaydet
    final_save_path = f"outputs/{args.dataset.lower()}_final_model"
    trainer.model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"\n✅ Eğitim Tamamlandı! Model kaydedildi: {final_save_path}")

if __name__ == "__main__":
    main()