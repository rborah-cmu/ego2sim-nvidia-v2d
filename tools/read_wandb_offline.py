"""Read the history of an OFFLINE W&B run (FlashCHORD training with TRAIN_LOG=offline) without any network access.

Usage (flash_chord venv, which has wandb): python read_wandb_offline.py RUN_OUT_DIR [--keys substr,substr] [--every N] [--list]
Finds OUT/wandb/offline-run-*/run-*.wandb, decodes its history records and prints a table of the selected metrics.
"""
import argparse
import glob
import json

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore


def history(path):
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            row = {}
            for item in rec.history.item:
                key = item.key or "/".join(item.nested_key)
                try:
                    row[key] = json.loads(item.value_json)
                except ValueError:
                    pass
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--keys", default="return,reward,termination,episode_length,voc,object")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.out_dir}/wandb/offline-run-*/run-*.wandb"))
    if not files:
        raise SystemExit(f"no offline run under {a.out_dir}/wandb")
    rows = history(files[-1])
    keys = sorted({k for r in rows for k in r})
    if a.list:
        print("\n".join(keys))
        return
    subs = [s for s in a.keys.split(",") if s]
    cols = [k for k in keys if any(s in k for s in subs) and not k.startswith("_")][:12]
    step_key = "_step"
    print(f"{len(rows)} records; columns: {cols}")
    print("step".rjust(11) + "".join(c[-22:].rjust(24) for c in cols))
    for r in rows[::a.every] + rows[-1:]:
        vals = "".join((f"{r[c]:.4g}" if isinstance(r.get(c), (int, float)) else "-").rjust(24) for c in cols)
        print(str(r.get(step_key, "")).rjust(11) + vals)


if __name__ == "__main__":
    main()
