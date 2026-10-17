#!/usr/bin/env bash
# Run ego2sim_public_episode.sh for "EP:OBJ1,OBJ2" items in sequence on one GPU; log per episode.
GPU=$1; shift
for item in "$@"; do EP=${item%%:*}; OBJS=${item#*:}
  echo "$(date +%T) start ep$EP ($OBJS)"; bash ~/v2d/scripts/ego2sim_public_episode.sh "$EP" "$OBJS" "$GPU" > ~/v2d/logs/e2s_ep$EP.log 2>&1; echo "$(date +%T) done ep$EP rc=$?"
done
