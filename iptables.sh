#!/bin/bash

# Database configuration
DB_NAME="ips"
DB_HOST="127.0.0.1"
DB_USER="firewall_user"
DB_PASS="achour"
TABLE="flows_meta"

export PGPASSWORD="$DB_PASS"

echo "[*] Starting dynamic firewall updater..."

while true; do
    # Get malicious IPs with prediction = 1
    MALICIOUS_IPS=$(psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -At -c \
        "SELECT DISTINCT srcip FROM $TABLE WHERE prediction = 1;")
    
    # Check if query returned results
    if [ -z "$MALICIOUS_IPS" ]; then
        echo "[*] No new malicious IPs to block. Checking again in 5 seconds..."
        sleep 5
        continue
    fi
    
    # Block each malicious IP
    while IFS= read -r IP; do
        # Skip empty lines
        if [ -z "$IP" ]; then
            continue
        fi
        
        # Block the IP using iptables
        if sudo iptables -A INPUT -s "$IP" -j DROP 2>/dev/null; then
            echo "[+] Blocked malicious IP: $IP"
            
            # Kill existing connections from this IP
            echo "[*] Killing existing connections from $IP..."
            sudo conntrack -D -s "$IP" 2>/dev/null
            
            # Alternative: use ss to find and kill connections
            sudo ss -K dst "$IP" 2>/dev/null
            
        else
            echo "[!] Failed to block IP: $IP"
        fi
    done <<< "$MALICIOUS_IPS"
    
    sleep 5
done
