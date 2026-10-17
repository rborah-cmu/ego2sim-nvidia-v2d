#!/usr/bin/env bash
for x in "41 0 white_pot -" "11 1 beige_cup 0" "13 1 plastic_dish_rack 0" "18 0 mini_sweeper 1" "21 1 water_pitcher 0" "23 0 wooden_piece_1 1"; do
  set -- $x; echo "=== ep$1 $3"; bash ~/v2d/scripts/e2s_make_capture.sh $1 $2 $3 $4 0 2>&1 | grep -v -i warn
done
