#!/usr/bin/env python3
import requests
import json

# Test masing-masing client
clients = [
    ("waru", 5001),
    ("kemiri", 5002),
    ("nangsri", 5003)
]

for name, port in clients:
    print(f"\n=== Client {name.upper()} ===")
    
    # Update model dengan dummy parameters
    resp = requests.post(
        f"http://localhost:{port}/api/update_model",
        json={'global_parameters': {'coef_': [[0]*7], 'intercept_': [0]}}
    )
    
    if resp.status_code == 200:
        result = resp.json()
        params = result.get('parameters', {})
        coef = params.get('coef_', [[]])
        
        print(f"  Status: OK")
        print(f"  Samples: {result.get('num_samples', 0)}")
        print(f"  Accuracy: {result.get('accuracy', 0)}")
        print(f"  Coef shape: {len(coef[0]) if coef else 0} features")
        print(f"  Coef values: {coef[0][:5]}..." if len(coef[0]) > 5 else f"  Coef: {coef}")
    else:
        print(f"  Status: FAILED ({resp.status_code})")
