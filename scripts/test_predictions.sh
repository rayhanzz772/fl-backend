#!/bin/bash

echo "Testing model predictions..."

# Ambil model dari server
curl -s http://localhost:5000/api/global_model > /tmp/global_model.json

echo "Global model parameters received"
echo ""

# Test prediksi dengan data dummy (1 sample)
echo "Testing prediction for a sample child:"

# Buat data test (feature values)
# Kirim ke client waru untuk prediksi
curl -s -X POST http://localhost:5001/api/update_model \
  -H "Content-Type: application/json" \
  -d '{
    "global_parameters": '$(cat /tmp/global_model.json | jq '.parameters')'
  }' | jq '.accuracy'

echo ""
echo "Untuk analisis lebih mendalam, perlu hitung:"
echo "1. Confusion Matrix (TP, TN, FP, FN)"
echo "2. F1-Score"
echo "3. Precision & Recall untuk kelas stunting"
