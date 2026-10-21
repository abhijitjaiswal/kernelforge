#!/bin/sh
# m5-test health log, one line per interval (default 60 s). ALERT lines on: memory free < 20 %, swap growth > 1 GB since start,
# disk free < 25 GB, thermal pressure not Nominal. Light: one top/vm sample + one 0.2 s powermetrics sample per interval.
INT=${1:-60}
LOG=$HOME/kernelforge/runs/health/health.log
mkdir -p "$(dirname "$LOG")"
swap0=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
echo "# start $(date '+%F %T') swap0=${swap0}MB interval=${INT}s" >> "$LOG"
while true; do
  t=$(date '+%F %T')
  free=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  pm=$(top -l 1 -n 0 | awk '/PhysMem/{print $2" used, "$4" wired, "$6" compressor"}' | tr -d '(),')
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
  disk=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  pw=$(sudo -n /usr/bin/powermetrics --samplers thermal,gpu_power -i 200 -n 1 2>/dev/null)
  therm=$(echo "$pw" | awk -F': ' '/pressure level/{print $2}')
  gpu=$(echo "$pw" | awk '/GPU HW active frequency/{print $5"MHz"} /GPU Power/{print $3"mW"}' | tr '\n' ' ')
  topp=$(ps -axo rss,comm | sort -rn | head -1 | awk '{printf "%s %.1fGB", $2, $1/1048576}' | sed 's#.*/##')
  echo "$t free=${free}% mem[$pm] swap=${swap}MB disk=${disk}GB thermal=${therm} gpu[${gpu}] top=${topp}" >> "$LOG"
  a=""
  [ "${free:-100}" -lt 20 ] && a="$a mem_free<20%"
  awk -v s="$swap" -v s0="$swap0" 'BEGIN{exit !(s-s0>1024)}' && a="$a swap+>1GB"
  [ "${disk:-999}" -lt 25 ] && a="$a disk<25GB"
  [ -n "$therm" ] && [ "$therm" != "Nominal" ] && a="$a thermal=$therm"
  [ -n "$a" ] && echo "$t ALERT$a" >> "$LOG"
  sleep "$INT"
done
