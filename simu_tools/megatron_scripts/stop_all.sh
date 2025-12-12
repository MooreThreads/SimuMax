#!/bin/bash

HOSTFILE=./hostfile
NUM_NODES=$(grep -v '^#\|^$' $HOSTFILE | wc -l)
echo "NUM_NODES: $NUM_NODES"

hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
    ssh -f -n $host "pkill -f '/opt/conda/envs/py310/bin/torchrun'"
    ssh -f -n $host "pkill -f '/usr/local/bin/torchrun'"
    echo "$host is killed."
done
