#!/bin/sh
# Run a command on m5-test in ~/kernelforge inside a detached tmux session (survives SSH drops).
# usage: scripts/remote.sh <job-name> '<command>'
#   log: m5-test:~/kernelforge/runs/jobs/<job-name>.log (ends with EXIT=<rc>)
#   attach on m5-test: /opt/homebrew/bin/tmux attach -t kf-<job-name>
name=$1; shift
printf 'cd "$HOME/kernelforge" || exit 1\n( %s ) > runs/jobs/%s.log 2>&1\necho EXIT=$? >> runs/jobs/%s.log\n' "$*" "$name" "$name" |
  ssh m5-test "mkdir -p ~/kernelforge/runs/jobs && cat > ~/kernelforge/runs/jobs/$name.sh && /opt/homebrew/bin/tmux new-session -d -s kf-$name 'sh ~/kernelforge/runs/jobs/$name.sh' && echo started kf-$name"
