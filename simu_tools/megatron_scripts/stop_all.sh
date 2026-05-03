#!/bin/bash

HOSTFILE=./hostfile
NUM_NODES=$(grep -v '^#\|^$' $HOSTFILE | wc -l)
echo "NUM_NODES: $NUM_NODES"

hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)

is_local_host() {
    local host="$1"
    local short_hostname
    short_hostname=$(hostname)
    local fqdn_hostname
    fqdn_hostname=$(hostname -f 2>/dev/null || hostname)
    [[ "$host" == "127.0.0.1" || "$host" == "localhost" || "$host" == "$short_hostname" || "$host" == "$fqdn_hostname" ]]
}

for host in ${hostlist[@]}; do
    if is_local_host "$host"; then
        pkill -f '/opt/conda/envs/py310/bin/torchrun' >/dev/null 2>&1 || true
        pkill -f '/usr/local/bin/torchrun' >/dev/null 2>&1 || true
    else
        ssh -f -n $host "pkill -f '/opt/conda/envs/py310/bin/torchrun'" || true
        ssh -f -n $host "pkill -f '/usr/local/bin/torchrun'" || true
    fi
    echo "$host is killed."
done
