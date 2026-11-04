#!/bin/bash
# Every 5 s, append the time, package temperature (°C), fan (RPM), mean CPU clock (MHz),
# package power (W) and 1-minute load to a CSV. Runs on the host, for sudo and its sensors.
# Usage: sample.sh <file.csv>
hwmon() { grep -l "^$1\$" /sys/class/hwmon/*/name | head -1 | xargs dirname; }
temp=$(hwmon coretemp)/temp1_input
fan=$(hwmon macsmc_hwmon)/fan1_input
energy=/sys/class/powercap/intel-rapl:0/energy_uj

[ -s "$1" ] || echo "time,temp_c,fan_rpm,clock_mhz,power_w,load1" >"$1"
last=$(sudo cat "$energy")
while sleep 5; do
  now=$(sudo cat "$energy")
  clock=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | awk '{s+=$1} END {printf "%d", s/NR/1000}')
  echo "$(date +%s),$(($(cat "$temp") / 1000)),$(cat "$fan"),$clock,$(((now - last) / 5000000)),$(cut -d' ' -f1 /proc/loadavg)" >>"$1"
  last=$now
done
