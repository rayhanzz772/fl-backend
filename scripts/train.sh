#!/bin/bash

# Training orchestration script
echo "Starting Federated Learning Training..."

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    echo "Starting services..."
    docker-compose up -d
    sleep 10
fi

# Run training orchestration
python3 << EOF
import requests
import time
import json

SERVER_URL = "http://localhost:5000"
CLIENTS = [
    {"name": "waru", "url": "http://localhost:5001"},
    {"name": "kemiri", "url": "http://localhost:5002"},
    {"name": "nangsri", "url": "http://localhost:5003"}
]

NUM_ROUNDS = 20

print("🚀 Starting Federated Learning Training")
print("="*50)

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n🔄 Round {round_num}/{NUM_ROUNDS}")
    
    # Get global model
    resp = requests.get(f"{SERVER_URL}/api/global_model")
    global_params = resp.json()['parameters']
    
    client_updates = []
    client_samples = []
    client_names = []
    client_accuracies = []
    
    # Train each client
    for client in CLIENTS:
        print(f"  Training {client['name']}...")
        resp = requests.post(
            f"{client['url']}/api/update_model",
            json={'global_parameters': global_params}
        )
        
        if resp.status_code == 200:
            result = resp.json()
            client_updates.append(result['parameters'])
            client_samples.append(result['num_samples'])
            client_names.append(client['name'])
            client_accuracies.append(result['accuracy'])
            print(f"    ✓ Accuracy: {result['accuracy']:.3f}")
    
    # Aggregate
    print("  Aggregating models...")
    requests.post(
        f"{SERVER_URL}/api/aggregate",
        json={
            'client_params': client_updates,
            'client_samples': client_samples,
            'client_names': client_names,
            'client_accuracies': client_accuracies
        }
    )
    
    time.sleep(1)

print("\n✅ Training completed!")
EOF