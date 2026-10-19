#!/bin/sh
# Run a command on m5-test in ~/kernelforge, detached with nohup (survives SSH drops).
# usage: scripts/remote.sh <job-name> <command...>   -> log at m5-test:~/kernelforge/runs/jobs/<job-name>.log
name=$1; shift
ssh m5-test "mkdir -p ~/kernelforge/runs/jobs && cd ~/kernelforge && nohup sh -c '$*; echo EXIT=\$?' > runs/jobs/$name.log 2>&1 < /dev/null & echo started $name"
