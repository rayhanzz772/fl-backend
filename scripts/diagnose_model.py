import requests
import numpy as np

def diagnose_client(port, name):
    print(f"\n{'='*40}")
    print(f"DIAGNOSING {name.upper()}")
    print(f"{'='*40}")
    
    # Get client info
    resp = requests.get(f"http://localhost:{port}/api/client_info")
    if resp.status_code == 200:
        info = resp.json()
        print(f"Total samples: {info['num_samples']}")
        print(f"Stunting rate: {info['stunting_rate']*100:.1f}%")
    
    # Get evaluation
    resp = requests.get(f"http://localhost:{port}/api/evaluate")
    if resp.status_code == 200:
        eval_data = resp.json()
        print(f"Model accuracy: {eval_data['accuracy']*100:.1f}%")
    
    print("\n⚠️  Jika akurasi model >> (100% - stunting_rate),")
    print("   kemungkinan model hanya memprediksi 'Normal' semua!")

# Diagnose all clients
diagnose_client(5001, "waru")
diagnose_client(5002, "kemiri")
diagnose_client(5003, "nangsri")

print("\n" + "="*40)
print("SOLUTION:")
print("="*40)
print("1. Gunakan class_weight='balanced' di LogisticRegression")
print("2. Atau oversampling data minoritas (stunting)")
print("3. Gunakan metrics F1-Score, bukan accuracy")
