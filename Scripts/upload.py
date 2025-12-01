import os
import argparse
from huggingface_hub import HfApi, login

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Bilgisayarındaki model klasörü (Örn: outputs/deep_final_model)")
    parser.add_argument("--repo_name", type=str, required=True, help="HuggingFace'te oluşacak isim (Örn: kullaniciadi/Qwen-Deep-LoRA)")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace Write Token (hf_...)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"\n🚀 MODEL YÜKLENİYOR...")
    print(f"Yerel Klasör: {args.model_path}")
    print(f"Hedef Repo:   {args.repo_name}")
    
    # 1. Giriş Yap
    try:
        login(token=args.token)
        print("✅ Hugging Face girişi başarılı!")
    except Exception as e:
        print(f"❌ Giriş hatası: {e}")
        return

    # 2. Yüklemeyi Başlat
    api = HfApi()
    
    try:
        # Repoyu oluştur (varsa hata vermez, devam eder)
        api.create_repo(repo_id=args.repo_name, exist_ok=True)
        
        # Dosyaları yükle
        print("-> Dosyalar buluta gönderiliyor (İnternet hızına göre sürer)...")
        api.upload_folder(
            folder_path=args.model_path,
            repo_id=args.repo_name,
            repo_type="model"
        )
        
        print(f"\n🎉 TEBRİKLER! Yükleme tamamlandı.")
        print(f"Linkin: https://huggingface.co/{args.repo_name}")
        
    except Exception as e:
        print(f"❌ Yükleme sırasında hata: {e}")

if __name__ == "__main__":
    main()
