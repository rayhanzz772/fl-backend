#!/usr/bin/env python3
import json

# Data dari hasil training
clients = {
    'waru': {'samples': 100, 'stunting_rate': 0.15, 'accuracy': 0.97},
    'kemiri': {'samples': 211, 'stunting_rate': 0.109, 'accuracy': 0.9716},
    'nangsri': {'samples': 141, 'stunting_rate': 0.092, 'accuracy': 0.9645}
}

print("="*60)
print("ANALISIS HASIL FEDERATED LEARNING")
print("="*60)

for name, data in clients.items():
    baseline = 1 - data['stunting_rate']  # if always predict "Normal"
    improvement = (data['accuracy'] - baseline) / baseline * 100
    
    print(f"\n📊 {name.upper()}:")
    print(f"   Stunting rate: {data['stunting_rate']*100:.1f}%")
    print(f"   Baseline accuracy (predict all Normal): {baseline*100:.1f}%")
    print(f"   Model accuracy: {data['accuracy']*100:.1f}%")
    print(f"   Improvement over baseline: {improvement:.1f}%")
    
    if improvement < 10:
        print(f"   ⚠️  Model hanya {improvement:.1f}% lebih baik dari tebak semua Normal!")
        print(f"   → Kemungkinan model TIDAK belajar mendeteksi stunting")

print("\n" + "="*60)
print("KESIMPULAN:")
print("="*60)
print("""
Model dengan akurasi 97% TIDAK BERARTI bagus untuk data imbalance!
Yang perlu dilihat:
1. F1-Score (keseimbangan precision & recall)
2. Recall (seberapa banyak stunting terdeteksi)
3. Precision (seberapa akurat prediksi stunting)

Tanpa F1-Score, accuracy 97% bisa dicapai hanya dengan:
"Memprediksi semua anak sebagai NORMAL" → 85-91% akurasi!
""")
