#!/bin/bash
while true; do
    clear
    echo "========================================="
    echo "FEDERATED LEARNING MONITOR"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="
    
    # Server status
    SERVER_STATUS=$(curl -s http://localhost:5000/health)
    echo "Server: $(echo $SERVER_STATUS | jq -r '.status // "unknown"')"
    echo "Rounds: $(echo $SERVER_STATUS | jq -r '.rounds // 0')"
    
    # Client status
    echo -e "\n--- Clients ---"
    for port in 5001 5002 5003; do
        CLIENT=$(curl -s http://localhost:$port/health 2>/dev/null)
        NAME=$(echo $CLIENT | jq -r '.client // "unknown"')
        STATUS=$(echo $CLIENT | jq -r '.status // "unknown"')
        SAMPLES=$(echo $CLIENT | jq -r '.samples // 0')
        echo "$NAME: $STATUS ($SAMPLES samples)"
    done
    
    sleep 3
done
