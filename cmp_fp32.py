import numpy as np, fx
from ref_fp32 import read_checkpoint, RunStateF, forward_fp32
from convert import load_a5
import oracle

cfg, w = read_checkpoint("synthetic260k.bin")
m = load_a5("synthetic260k.a5")
sf, si = RunStateF(cfg), oracle.State(m)
toks = [1,5,9,42,7,100,3,55]
for p,t in enumerate(toks):
    a = forward_fp32(cfg, w, sf, t, p)
    b = np.array(oracle.forward(m, si, t, p), dtype=np.float64)/fx.ONE
    r = np.corrcoef(a,b)[0,1]
    rel = np.abs(a-b).max()/ (np.abs(a).max()+1e-9)
    ov = len(set(np.argsort(-a)[:10]) & set(np.argsort(-b)[:10]))
    print(f"pos {p}: corr={r:.5f}  max_rel_err={rel:.4f}  top10_overlap={ov}/10")
