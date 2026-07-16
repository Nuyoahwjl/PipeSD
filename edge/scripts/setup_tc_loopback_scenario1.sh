#!/usr/bin/env bash
set -euo pipefail

# Shape a cloud server reached through 127.0.0.1.  Requests whose destination
# port is SERVER_PORT are the 2.5 MB/s uplink; responses whose source port is
# SERVER_PORT are the 25 MB/s downlink.
DEV="${DEV:-lo}"
SERVER_PORT="${SERVER_PORT:-8000}"
UPLINK_MBIT="${UPLINK_MBIT:-20}"
DOWNLINK_MBIT="${DOWNLINK_MBIT:-200}"

if [[ "${1:-}" == "--clear" ]]; then
  sudo tc qdisc del dev "$DEV" root 2>/dev/null || true
  exit 0
fi

sudo tc qdisc replace dev "$DEV" root handle 1: htb default 30
sudo tc class replace dev "$DEV" parent 1: classid 1:10 htb rate "${UPLINK_MBIT}mbit" ceil "${UPLINK_MBIT}mbit"
sudo tc class replace dev "$DEV" parent 1: classid 1:20 htb rate "${DOWNLINK_MBIT}mbit" ceil "${DOWNLINK_MBIT}mbit"
sudo tc class replace dev "$DEV" parent 1: classid 1:30 htb rate 1000mbit ceil 1000mbit
sudo tc filter replace dev "$DEV" protocol ip parent 1: prio 1 u32 \
  match ip dport "$SERVER_PORT" 0xffff flowid 1:10
sudo tc filter replace dev "$DEV" protocol ip parent 1: prio 2 u32 \
  match ip sport "$SERVER_PORT" 0xffff flowid 1:20

tc -s class show dev "$DEV"
