#!/usr/bin/env python3
import requests
import time
import json
import sys

SERVER_URL = "http://localhost:5000"
CLIENTS = [
    {"name": "waru", "url": "http://localhost:5001"},
    {"name": "kemiri", "url": "http://localhost:5002"},
    {"name": "nangsri", "url": "http://localhost:5003"}
]

NUM_ROUNDS = 20
LOCAL_EPOCHS = 5

print("="*60)
print("🚀 FEDERATED LEARNING TRAINING WITH REAL DATA")
print("="*60)
print(f"Total Rounds: {NUM_ROUNDS}")
print(f"Local Epochs per Round: {LOCAL_EPOCHS}")
print(f"Total Clients: {len(CLIENTS)}")
print("="*60)

# Get initial client info
print("\n📊 Client Data Summary:")
for client in CLIENTS:
    resp = requests.get(f"{client['url']}/api/client_info")
    if resp.status_code == 200:
        info = resp.json()
        print(f"  {client['name'].upper()}: {info['num_samples']} samples, stunting rate: {info['stunting_rate']*100:.1f}%")
print("="*60)

# Store history
history = {
    'rounds': [],
    'global_accuracies': [],
    'client_accuracies': {c['name']: [] for c in CLIENTS}
}

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n🔄 ROUND {round_num}/{NUM_ROUNDS}")
    print("-" * 50)
    
    # 1. Get global model
    try:
        resp = requests.get(f"{SERVER_URL}/api/global_model", timeout=5)
        if resp.status_code != 200:
            print(f"  ❌ Failed to get global model")
            continue
        global_params = resp.json().get('parameters', {})
        print(f"  📥 Global model received")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue
    
    # 2. Train each client locally
    client_updates = []
    client_samples = []
    client_names = []
    client_accuracies = []
    
    for client in CLIENTS:
        print(f"\n  📍 Training {client['name'].upper()}...", end=" ")
        
        try:
            start = time.time()
            resp = requests.post(
                f"{client['url']}/api/update_model",
                json={'global_parameters': global_params},
                timeout=60
            )
            elapsed = time.time() - start
            
            if resp.status_code == 200:
                result = resp.json()
                client_updates.append(result['parameters'])
                client_samples.append(result['num_samples'])
                client_names.append(client['name'])
                acc = result.get('accuracy', 0)
                client_accuracies.append(acc)
                history['client_accuracies'][client['name']].append(acc)
                print(f"✓ acc={acc:.3f} ({elapsed:.1f}s)")
            else:
                print(f"❌ HTTP {resp.status_code}")
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
    
    # 3. Aggregate on server
    if len(client_updates) >= 2:
        print(f"\n  🔄 Aggregating {len(client_updates)} clients...")
        
        try:
            resp = requests.post(
                f"{SERVER_URL}/api/aggregate",
                json={
                    'client_params': client_updates,
                    'client_samples': client_samples,
                    'client_names': client_names
                },
                timeout=30
            )
            
            if resp.status_code == 200:
                result = resp.json()
                avg_acc = sum(client_accuracies) / len(client_accuracies) if client_accuracies else 0
                history['rounds'].append(round_num)
                history['global_accuracies'].append(avg_acc)
                
                print(f"  ✅ Round {result['round']} completed")
                print(f"  📊 Average Accuracy: {avg_acc:.3f}")
                
                # Progress bar
                progress = round_num / NUM_ROUNDS
                bar = '█' * int(30 * progress) + '░' * (30 - int(30 * progress))
                print(f"  Progress: |{bar}| {round_num}/{NUM_ROUNDS}")
            else:
                print(f"  ❌ Aggregation failed")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    else:
        print(f"\n  ⚠️ Not enough clients responded")
    
    time.sleep(1)

print("\n" + "="*60)
print("✅ TRAINING COMPLETED!")
print("="*60)

# Final summary
print("\n📊 TRAINING SUMMARY")
print("-" * 40)
print(f"Total Rounds Completed: {len(history['rounds'])}/{NUM_ROUNDS}")

if history['global_accuracies']:
    print(f"Final Global Accuracy: {history['global_accuracies'][-1]:.3f}")
    print(f"Best Global Accuracy: {max(history['global_accuracies']):.3f}")

print("\n📈 Per Client Final Accuracy:")
for client in CLIENTS:
    acc_history = history['client_accuracies'][client['name']]
    if acc_history:
        print(f"  {client['name'].upper()}: {acc_history[-1]:.3f} (avg: {sum(acc_history)/len(acc_history):.3f})")

# Get final server status
print("\n🔍 Final Server Status:")
try:
    resp = requests.get(f"{SERVER_URL}/api/training_status")
    if resp.status_code == 200:
        status = resp.json()
        print(f"  Total rounds on server: {status.get('current_round', 0)}")
except:
    pass

# Save results
with open('training_results_real.json', 'w') as f:
    json.dump(history, f, indent=2)
print(f"\n💾 Results saved to: training_results_real.json")
