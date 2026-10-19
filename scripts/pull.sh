#!/bin/sh
# Pull results back from m5-test (runs + baselines; additive, never deletes locally).
cd "$(dirname "$0")/.." && rsync -az m5-test:~/kernelforge/runs/ ./runs/ && rsync -az m5-test:~/kernelforge/baselines/ ./baselines/
