#!/usr/bin/env bash
# DS4 rig GPU power/clock setup (needs sudo). Not persistent across reboots.
#
# PL 270 W: +27% prefill vs 220 W at 130K depth (DS4HANDOFF 0e power curve).
# -lgc 1995: +8.8% decode — removes DVFS droop between layer-split pipeline
# windows; 1695 does nothing, 2130 adds nothing over 1995, memory clock lock
# is a no-op. See docs/ds4/DS4_H0_LOCK_CLOCKS_2026-08-05.md.
set -e
sudo nvidia-smi -pl 270
sudo nvidia-smi -lgc 1995
nvidia-smi --query-gpu=index,power.limit,clocks.sm --format=csv
