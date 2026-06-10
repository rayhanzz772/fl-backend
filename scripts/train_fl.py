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

NUM_ROUNDS = 10
LOCAL_EPOCHS = 3

def print_progress(round_num, total, client_results):
    progress = round_num / total
    bar_length = 30
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    sys.stdout.write(f"\rProgress: |{bar}| {round_num}/{total} rounds")
    sys.stdout.flush()

print("="*60)
print("🚀 FEDERATED LEARNING TRAINING")
print("="*60)
print(f"Total Rounds: {NUM_ROUNDS}")
print(f"Local Epochs per Round: {LOCAL_EPOCHS}")
print(f"Total Clients: {len(CLIENTS)}")
print("="*60)

# Store history
history = {
    'rounds': [],
    'accuracies': [],
    'client_accuracies': {c['name']: [] for c in CLIENTS}
}

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n\n🔄 ROUND {round_num}/{NUM_ROUNDS}")
    print("-" * 50)
    
    # 1. Get global model from server
    try:
        resp = requests.get(f"{SERVER_URL}/api/global_model", timeout=5)
        if resp.status_code != 200:
            print(f"  ❌ Failed to get global model: {resp.status_code}")
            continue
        global_params = resp.json().get('parameters', {})
        print(f"  📥 Global model received")
    except Exception as e:
        print(f"  ❌ Error getting global model: {str(e)[:50]}")
        continue
    
    # 2. Train each client locally
    client_updates = []
    client_samples = []
    client_names = []
    client_accuracies = []
    
    for client in CLIENTS:
        print(f"\n  📍 Training {client['name'].upper()}...")
        
        try:
            start_time = time.time()
            resp = requests.post(
                f"{client['url']}/api/update_model",
                json={'global_parameters': global_params},
                timeout=60
            )
            elapsed = time.time() - start_time
            
            if resp.status_code == 200:
                result = resp.json()
                client_updates.append(result['parameters'])
                client_samples.append(result['num_samples'])
                client_names.append(client['name'])
                accuracy = result.get('accuracy', 0)
                client_accuracies.append(accuracy)
                history['client_accuracies'][client['name']].append(accuracy)
                print(f"     ✓ Accuracy: {accuracy:.3f} ({elapsed:.1f}s)")
            else:
                print(f"     ❌ Failed: HTTP {resp.status_code}")
        except Exception as e:
            print(f"     ❌ Error: {str(e)[:50]}")
    
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
                avg_accuracy = sum(client_accuracies) / len(client_accuracies) if client_accuracies else 0
                history['rounds'].append(round_num)
                history['accuracies'].append(avg_accuracy)
                
                print(f"  ✅ Round {result['round']} completed")
                print(f"  📊 Average Accuracy: {avg_accuracy:.3f}")
            else:
                print(f"  ❌ Aggregation failed: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  ❌ Aggregation error: {str(e)[:50]}")
    else:
        print(f"\n  ⚠️ Not enough clients responded (need at least 2)")
    
    print_progress(round_num, NUM_ROUNDS, client_accuracies)
    time.sleep(2)

print("\n\n" + "="*60)
print("✅ TRAINING COMPLETED!")
print("="*60)

# Final summary
print("\n📊 TRAINING SUMMARY")
print("-" * 40)
print(f"Total Rounds Completed: {len(history['rounds'])}/{NUM_ROUNDS}")
print(f"Final Global Accuracy: {history['accuracies'][-1] if history['accuracies'] else 0:.3f}")

print("\n📈 Per Client Final Accuracy:")
for client in CLIENTS:
    acc_history = history['client_accuracies'][client['name']]
    if acc_history:
        print(f"  {client['name'].upper()}: {acc_history[-1]:.3f} (avg: {sum(acc_history)/len(acc_history):.3f})")

# Get final status
print("\n🔍 Checking final server status...")
try:
    resp = requests.get(f"{SERVER_URL}/api/training_status")
    if resp.status_code == 200:
        status = resp.json()
        print(f"  Server rounds: {status.get('current_round', 0)}")
        print(f"  Features: {status.get('n_features', 0)}")
except:
    pass

# Save results
with open('training_results.json', 'w') as f:
    json.dump(history, f, indent=2)
print(f"\n💾 Results saved to: training_results.json")
