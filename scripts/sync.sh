#!/bin/sh
# Push local code to m5-test:~/kernelforge (never touches runs/ or models/ there).
cd "$(dirname "$0")/.." && rsync -az --delete --exclude .git --exclude runs --exclude models --exclude baselines --exclude baselines_video --exclude '__pycache__' ./ m5-test:~/kernelforge/
