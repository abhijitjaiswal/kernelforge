#!/bin/sh
# Pull results back from m5-test (runs + baselines; additive, never deletes locally).
cd "$(dirname "$0")/.." && ssh m5-test 'mkdir -p ~/kernelforge/runs ~/kernelforge/baselines' && \
  rsync -az m5-test:~/kernelforge/runs/ ./runs/ && rsync -az m5-test:~/kernelforge/baselines/ ./baselines/ && rsync -az --exclude "*.safetensors" m5-test:~/kernelforge/baselines_video/ ./baselines_video/
