# 10_protocol_ladder - Cell 65: helpers for the figures and tables of the results that have none yet (read-only). Uses the notebook's report() from Cell 1, so every new figure has the format of
# the existing ones (ROC of the models with the trivial baselines dashed; confusion matrices summed over seeds; table = mean over seeds (min-max)). Run once per kernel, before Cells 66-70.
import contextlib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from circmtl import ladder as L
assert all(n in globals() for n in ("report", "gd", "D", "OUT", "OLD", "np", "pd")), "run Cell 0 and Cell 1 first (they define report, OUT, OLD and gd)"
assert callable(report), "report has been overwritten; rerun Cell 1"
ROUND1 = D / "train_results_round1"
BL_HOST = ("disease_degree", "circrna_degree", "log_degree_sum")                                                                   # as in the S0 and B reports
BL_PAIR = ("disease_degree", "circrna_degree", "circrna_inverse_degree", "log_degree_sum", "kernel_circ", "kernel_dis", "kernel_both", "shortcut_logit")      # as in Cells 13, 16 and 19
nC65, nD65 = gd.N["circRNA"], gd.N["disease"]

def _bs_strict(d, kind):
    """popularity baselines of the host-gene folds, from the fold's TRAINING pairs only (as Cell 1)"""
    out = np.zeros(len(d))
    for (s, k), ix in d.groupby(["seed", "fold"]).indices.items():
        tr = gd.load_fold(int(s), int(k))["disease_train"]; dd = np.bincount(tr[:, 1], minlength=nD65); cd = np.bincount(tr[:, 0], minlength=nC65)
        c, t = d["c"].to_numpy()[ix], d["t"].to_numpy()[ix]
        out[ix] = {"disease_degree": dd[t], "circrna_degree": cd[c], "log_degree_sum": np.log1p(dd[t]) + np.log1p(cd[c])}[kind]
    return out

def _bs_pair_factory(tag):
    def _bs_pair(d, kind):                                                           # as Cell 13: baselines of the pair-level folds from TRAINING pairs only
        out = np.zeros(len(d))
        for (s, k), ix in d.groupby(["seed", "fold"]).indices.items():
            fd = L.pair_level_fold(gd, int(s), int(k)); Y = L.label_matrix(fd["disease_train"], nC65, nD65); dd, dc = Y.sum(0), Y.sum(1)
            c, t = d["c"].to_numpy()[ix], d["t"].to_numpy()[ix]
            if kind == "disease_degree": sc = dd[t]
            elif kind == "circrna_degree": sc = dc[c]
            elif kind == "circrna_inverse_degree": sc = -dc[c]
            elif kind == "log_degree_sum": sc = np.log1p(dd[t]) + np.log1p(dc[c])
            elif kind.startswith("kernel_"): sc = L.kernel_scores(Y)[kind][c, t]
            elif kind == "shortcut_logit":                                           # fitted on validation only
                z = np.load(OUT / "frozen_pair" / f"neg{tag}_seed{int(s)}_fold{int(k)}.npz")
                pv = np.vstack([fd["disease_val"], z["neg_val"]]); yv_ = np.r_[np.ones(len(fd["disease_val"])), np.zeros(len(z["neg_val"]))]
                clf = LogisticRegression(C=1.0).fit(np.c_[np.log1p(dd[pv[:, 1]]), np.log1p(dc[pv[:, 0]])], yv_)
                sc = clf.decision_function(np.c_[np.log1p(dd[t]), np.log1p(dc[c])])
            else: raise ValueError(kind)
            out[ix] = sc
        return out
    return _bs_pair

def _bs_external(d, kind):
    """run X: new circRNAs have no training pairs; the popularity baseline uses the dataset_v1 degrees of the diseases (as Cell 49)"""
    dd = np.bincount(gd.E["disease"][:, 1], minlength=nD65).astype(float)[d["t"].to_numpy()]
    return {"disease_degree": dd, "circrna_degree": np.zeros(len(d)), "log_degree_sum": np.log1p(dd)}[kind]

@contextlib.contextmanager
def baselines_for(kind, tag=None):
    """report() looks up the global name baseline_scores; swap it for the right protocol and ALWAYS restore it"""
    keep = globals()["baseline_scores"] if "baseline_scores" in globals() else None
    globals()["baseline_scores"] = {"strict": _bs_strict, "pair": _bs_pair_factory(tag) if kind == "pair" else None, "external": _bs_external}[kind]
    try: yield
    finally:
        if keep is not None: globals()["baseline_scores"] = keep

def read_arm(folder, model, label=None, pattern=None):
    """all prediction files of one model in one folder; the model column is relabelled when a label is given"""
    files = sorted(Path(folder).glob(pattern or f"pred_{model}_seed*.csv")); assert files, f"no prediction files for {model} in {folder}"
    P = pd.concat([pd.read_csv(f) for f in files], ignore_index=True); P = P[P["model"] == model].copy(); assert len(P), f"model {model} not in the files of {folder}"
    if label: P["model"] = label
    return P

def run_report(P, models, tag, scheme, kind, ptag=None, baselines=None):
    """the notebook's report() with the right baselines; handles files without a fold column (run X) or without score_rank"""
    P = P.copy()
    if "fold" not in P: P["fold"] = 0
    if "score_rank" not in P: P["score_rank"] = P.groupby(["model", "seed", "fold", "task", "scheme"])["score"].rank(pct=True)
    if "threshold" in P:
        has = P.groupby("model")["threshold"].apply(lambda s: bool(s.notna().all())); assert has.all() or not has.any(), "some models have thresholds and others not: do not mix them in one figure"
    with baselines_for(kind, ptag):
        return report(P, list(models), tag=tag, scheme=scheme, baselines=baselines or (BL_PAIR if kind == "pair" else BL_HOST))
print("Cell 65 ok: helpers defined (read_arm, run_report, baselines_for); figures and tables go to", OUT, "as report_<tag>_<scheme>.png and .csv")
