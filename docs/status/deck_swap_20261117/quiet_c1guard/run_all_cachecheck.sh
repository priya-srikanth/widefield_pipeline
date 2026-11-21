#!/usr/bin/env bash
# Whole quiet chain for the cache check, in the order the 11-18 sync-end run used (v2; then fr / task / hemo; tasksumm
# after task; eta / still / examples / movie after fr; stillplot after still). Prints per-step wall time.
R=$(dirname "$0")/run_chain_cachecheck.sh
t() { local s=$(date +%s); bash "$R" "$@"; echo "step $1 exit $? $(( $(date +%s) - s )) s"; }
T0=$(date +%s)
t v2 3
t fr 3 & P1=$!
( t task 3; t tasksumm ) & P2=$!
t hemo 3 & P3=$!
wait $P1
t eta 3 & P4=$!
( t still 3; t stillplot ) & P5=$!
t examples & P6=$!
t movie & P7=$!
wait $P2 $P3 $P4 $P5 $P6 $P7
echo "chain total $(( $(date +%s) - T0 )) s"
