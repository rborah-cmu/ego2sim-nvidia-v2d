"""Copy an official evaluator parquet with achieved := reference (scores the reconstruction alone, no RL)."""
import sys
import pyarrow as pa, pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
for c in [c for c in t.column_names if c.startswith("achieved_")]:
    t = t.set_column(t.column_names.index(c), c, t.column("reference_" + c[len("achieved_"):]))
pq.write_table(t, sys.argv[2])
