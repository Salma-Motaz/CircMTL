"""Reproduce the no-model floor on circ2DGNN's own shipped test set.
Usage:  python scripts/verify_circ2dgnn_floor.py <path-to-cloned-repo>      (clone: git clone https://github.com/CEN-KL/circ2DGNN)
Needs numpy, scipy, scikit-learn. Reads only files shipped in the repository; trains nothing.
"""
import pickle, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score

root = Path(sys.argv[1]) / "data" if len(sys.argv) > 1 else Path("circ2DGNN/data")
L = lambda f: sp.load_npz(root / "adj_matrix" / f).tocsr()
full, tr, te = L("01_csc_circ_disease.npz"), L("16_csc_circ_disease_train.npz"), L("17_csc_circ_disease_test.npz")
ms = pickle.load(open(root / "heterogeneous_graphs" / "mark_src.pkl", "rb"))
n_circ, n_dis = full.shape
assert ((tr + te) != full).nnz == 0, "train + test is not the full matrix"
pu, pv = te.nonzero()
nu, nv = [], []
for u in range(n_circ):                                   # mark_src[u] = known positives of u + the test negatives drawn for u
    for v in set(ms[u]) - set(full[u].indices):
        nu.append(u); nv.append(v)
nu, nv = np.array(nu), np.array(nv)
assert len(nu) == len(pu), f"recovered {len(nu)} negatives for {len(pu)} test positives"
y = np.r_[np.ones(len(pu)), np.zeros(len(nu))]; U = np.r_[pu, nu]; V = np.r_[pv, nv]
deg_d = np.asarray(tr.sum(0)).ravel().astype(float)       # disease popularity from the TRAINING positives only
deg_c = np.asarray(tr.sum(1)).ravel().astype(float)
print(f"dataset: {n_circ} circRNAs, {n_dis} diseases, {full.nnz} positive pairs | train {tr.nnz} | test {te.nnz} positives + {len(nu)} negatives")
print(f"diseases with any training positive: {int((deg_d > 0).sum())} of {n_dis}")
print(f"test negatives whose disease has no training positive: {np.mean(deg_d[nv] == 0):.1%} | test positives: {np.mean(deg_d[pv] == 0):.1%}")
print(f"\nNO-MODEL RULES on the shipped test set")
print(f"  disease popularity (training positives)  AUROC {roc_auc_score(y, deg_d[V]):.4f}  AUPR {average_precision_score(y, deg_d[V]):.4f}")
print(f"  circRNA training degree, reversed        AUROC {roc_auc_score(y, -deg_c[U]):.4f}")
print(f"\nreported in the paper for circ2DGNN: AUC 0.9779, AUPR 0.9803, F1 0.9309")
