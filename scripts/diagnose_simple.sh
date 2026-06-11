#!/bin/bash

echo "========================================="
echo "FEDERATED LEARNING DIAGNOSIS"
echo "========================================="

echo -e "\n📊 CLIENT DATA SUMMARY:"
echo "------------------------"
for port in 5001 5002 5003; do
    NAME=$(curl -s http://localhost:$port/api/client_info | python3 -c "import sys, json; print(json.load(sys.stdin).get('name', 'unknown'))" 2>/dev/null)
    SAMPLES=$(curl -s http://localhost:$port/api/client_info | python3 -c "import sys, json; print(json.load(sys.stdin).get('num_samples', 0))" 2>/dev/null)
    RATE=$(curl -s http://localhost:$port/api/client_info | python3 -c "import sys, json; print(json.load(sys.stdin).get('stunting_rate', 0))" 2>/dev/null)
    echo "$NAME: $SAMPLES samples, stunting rate: $(echo "$RATE * 100" | bc)%"
done

echo -e "\n📈 MODEL EVALUATION:"
echo "------------------------"
for port in 5001 5002 5003; do
    NAME=$(curl -s http://localhost:$port/api/client_info | python3 -c "import sys, json; print(json.load(sys.stdin).get('name', 'unknown'))" 2>/dev/null)
    ACC=$(curl -s http://localhost:$port/api/evaluate | python3 -c "import sys, json; print(json.load(sys.stdin).get('accuracy', 0))" 2>/dev/null)
    echo "$NAME: Accuracy = $(echo "$ACC * 100" | bc)%"
done

echo -e "\n🎯 INTERPRETATION:"
echo "------------------------"
echo "Jika Accuracy >> (100% - stunting_rate), kemungkinan model bias ke kelas mayoritas"
echo "Contoh: Stunting rate 10% → baseline accuracy 90% (jika selalu prediksi normal)"
echo "         Jika model accuracy 97% → hanya 7% lebih baik dari baseline"
echo ""
echo "Seharusnya:"
echo "  - Accuracy: 70-85% (realistis)"
echo "  - F1-Score: 0.5-0.7 (lebih bermakna untuk imbalance data)"
