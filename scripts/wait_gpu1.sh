#!/usr/bin/env bash
# exit 0 when our preprocessing sessions on GPU 1 are finished
for s in eval_pre1 eval_pre2 ep35cap; do tmux has-session -t $s 2>/dev/null && exit 1; done; exit 0