cd ~/v2d/agenticSTAR
R=$(ls -dt runs/${E2S_TAG:-ep2_white_pot_lid}_* 2>/dev/null | head -1)
[ -z "$R" ] && { echo "norun"; exit 0; }
L=$R/supervisor/combined.log
it=$(ls $R/iterations 2>/dev/null | wc -l)
age=$(( $(date +%s) - $(stat -c %Y $L 2>/dev/null || echo 0) ))
procs=$(pgrep -f "claude-supervisor|claude-articulated" | wc -l)
st=$(python3 -c "import json;print(json.load(open('$R/supervisor/state.json')).get('status'))" 2>/dev/null || echo ?)
cs=$(python3 -c "import json;d=json.load(open('$R/contract_status.json'));print('allpass' if d['all_pass'] else 'fail:'+','.join(k for k,v in d['checks'].items() if not v['passes']))" 2>/dev/null || echo none)
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
echo "run=$(basename $R) iters=$it procs=$procs status=$st contract=$cs gpu0MiB=$gpu logage=$age"
