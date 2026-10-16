"""Protocol-ladder helpers (numpy only): frozen uniform-pair negatives, uniform training sampler,
leak assertions and the validation-chosen F1 threshold. Rules: configs/protocol_ladder.yaml.
Does not touch the frozen strict-protocol artefacts."""
from __future__ import annotations

import numpy as np


def codes(c, t, n_t):
    return np.asarray(c, np.int64) * n_t + np.asarray(t, np.int64)


def sample_uniform_pairs(circ_pool, n, n_dis, exclude_codes, rng, max_rounds=200):
    """n UNIQUE pairs: circRNA uniform over circ_pool, disease uniform over all diseases, never in exclude_codes."""
    pool, out = np.asarray(circ_pool, np.int64), np.empty(0, np.int64)
    for _ in range(max_rounds):
        need = n - len(out)
        if need <= 0:
            break
        m = 3 * need + 16
        cand = codes(rng.choice(pool, size=m), rng.integers(0, n_dis, size=m), n_dis)
        cand = cand[~np.isin(cand, exclude_codes)]
        cand = cand[~np.isin(cand, out)]
        _, first = np.unique(cand, return_index=True)
        out = np.concatenate([out, cand[np.sort(first)]])[:n]
    assert len(out) == n, "could not draw enough unique negatives"
    return np.c_[out // n_dis, out % n_dis].astype(np.int64)


def frozen_uniform_negatives(gd, fd, seed, fold):
    """Run B: one uniform-pair negative per positive. Validation circRNAs for validation, test circRNAs for test.
    Never a pair that is a positive anywhere in the full matrix."""
    n_dis, known = gd.N["disease"], gd.known_codes["disease"]
    neg_val = sample_uniform_pairs(fd["val"], len(fd["disease_val"]), n_dis, known, np.random.default_rng(30_000 + 1000 * seed + fold))
    neg_test = sample_uniform_pairs(fd["test"], len(fd["disease_test"]), n_dis, known, np.random.default_rng(40_000 + 1000 * seed + fold))
    return neg_val, neg_test


def uniform_train_negatives(n_dis, pos, pool, exclude_codes, rng, k=5):
    """Training negatives: k per supervision edge; circRNA uniform over the TRAINING circRNAs (pool), disease uniform;
    excludes TRAINING positives only (never reads validation or test labels)."""
    pool = np.asarray(pool, np.int64)
    c = rng.choice(pool, size=len(pos) * k)
    t = rng.integers(0, n_dis, size=len(c))
    for _ in range(50):
        bad = np.isin(codes(c, t, n_dis), exclude_codes)
        if not bad.any():
            break
        c[bad], t[bad] = rng.choice(pool, size=int(bad.sum())), rng.integers(0, n_dis, size=int(bad.sum()))
    keep = ~np.isin(codes(c, t, n_dis), exclude_codes)
    return np.c_[c[keep], t[keep]].astype(np.int64)


def check_fold_B(gd, fd, neg_val, neg_test):
    """Leak assertions of protocol_ladder.yaml for a host-gene fold with uniform negatives. Raises AssertionError."""
    n_dis, known = gd.N["disease"], gd.known_codes["disease"]
    tr, va, te = (set(map(int, fd[k])) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te), "circRNA sets of train/val/test overlap"
    for part, s in (("train", tr), ("val", va), ("test", te)):
        assert set(map(int, fd[f"disease_{part}"][:, 0])) <= s, f"disease_{part} has a circRNA outside {part}"
    tc = codes(fd["disease_train"][:, 0], fd["disease_train"][:, 1], n_dis)
    for part in ("val", "test"):
        pc = codes(fd[f"disease_{part}"][:, 0], fd[f"disease_{part}"][:, 1], n_dis)
        assert not np.isin(pc, tc).any(), f"a {part} positive is also a training edge"
    for name, neg, s, npos in (("neg_val", neg_val, va, len(fd["disease_val"])), ("neg_test", neg_test, te, len(fd["disease_test"]))):
        nc = codes(neg[:, 0], neg[:, 1], n_dis)
        assert len(neg) == npos, f"{name}: {len(neg)} negatives for {npos} positives"
        assert set(map(int, neg[:, 0])) <= s, f"{name}: circRNA outside its own set"
        assert not np.isin(nc, known).any(), f"{name}: contains a known positive"
        assert len(np.unique(nc)) == len(nc), f"{name}: duplicate pairs"
    return True


def check_train_negatives(n_dis, neg, fd):
    """The training sampler used only training circRNAs and never excluded a validation or test label."""
    assert set(map(int, neg[:, 0])) <= set(map(int, fd["train"])), "training negative uses a non-training circRNA"
    tc = codes(fd["disease_train"][:, 0], fd["disease_train"][:, 1], n_dis)
    assert not np.isin(codes(neg[:, 0], neg[:, 1], n_dis), tc).any(), "training negative equals a training positive"
    return True


def best_f1_threshold(y, s):
    """Score cut that maximises F1 on (y, s); predict positive when score > threshold. Returns (threshold, F1)."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp, fp, npos = np.cumsum(ys), np.cumsum(1 - ys), ys.sum()
    f1 = 2 * tp / (tp + fp + npos)
    cut = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    i = cut[np.argmax(f1[cut])]
    thr = (ss[i] + ss[i + 1]) / 2 if i + 1 < len(ss) else ss[i] - 1e-9
    return float(thr), float(f1[i])


# ====================================================================== pair-level protocol (runs A and C)
AUX = ("mirna", "rbp")


def pair_level_fold(gd, seed, fold, n_folds=5, val_frac=0.2):
    """Pair-level split: one random permutation of ALL known circRNA-disease pairs per seed, cut into n_folds folds
    (every pair is in exactly one test fold); validation = val_frac of the training pairs. Auxiliary (ENCORI) edges are not
    disease labels: all of them stay in 'train' for every circRNA. Returns a dict that GraphData's consumers accept."""
    E = gd.E["disease"]
    n = len(E)
    perm = np.random.default_rng(50_000 + seed).permutation(n)
    fold_of = np.empty(n, np.int64)
    fold_of[perm] = np.arange(n) % n_folds
    test_i, rest = np.flatnonzero(fold_of == fold), np.flatnonzero(fold_of != fold)
    n_val = int(round(val_frac * len(rest)))
    sel = np.random.default_rng(60_000 + 1000 * seed + fold).permutation(len(rest))
    val_i, train_i = np.sort(rest[sel[:n_val]]), np.sort(rest[sel[n_val:]])
    tr, va, te = E[train_i], E[val_i], E[test_i]
    train_c = np.unique(tr[:, 0])                                   # circRNAs with at least one training pair
    xt = gd.X_raw[train_c].astype(np.float64)
    fd = {"seed": seed, "fold": fold, "train": train_c, "val": np.unique(va[:, 0]), "test": np.unique(te[:, 0]),
          "mu": xt.mean(0), "sd": xt.std(0) + 1e-8, "disease_train": tr, "disease_val": va, "disease_test": te}
    for r in AUX:
        fd[f"{r}_train"], fd[f"{r}_val"], fd[f"{r}_test"] = gd.E[r], gd.E[r][:0], gd.E[r][:0]
    return fd


def degree_matched_negatives(pos, deg, n_dis, known_codes, rng, max_rounds=200):
    """Same circRNA as each positive, disease ~ training degree, never a known positive, no duplicates (as in the strict protocol)."""
    p = deg / deg.sum()
    c, t = pos[:, 0].copy(), rng.choice(n_dis, size=len(pos), p=p)
    for _ in range(max_rounds):
        cd = codes(c, t, n_dis)
        _, first = np.unique(cd, return_index=True)
        dup = np.ones(len(cd), bool); dup[first] = False
        bad = np.isin(cd, known_codes) | dup
        if not bad.any():
            break
        t[bad] = rng.choice(n_dis, size=int(bad.sum()), p=p)
    assert not bad.any(), "could not draw degree-matched negatives without duplicates"
    return np.c_[c, t].astype(np.int64)


def frozen_negatives_pair(gd, fd, seed, fold, scheme):
    """Frozen validation and test negatives for a pair-level fold. scheme: 'degree_matched' (run A), 'uniform_pair' (run C) or 'uniform_test_circ' (run C2)."""
    n_dis, known = gd.N["disease"], gd.known_codes["disease"]
    deg = np.bincount(fd["disease_train"][:, 1], minlength=n_dis).astype(float)
    out = []
    for part, base in (("val", 70_000), ("test", 80_000)):
        pos, rng = fd[f"disease_{part}"], np.random.default_rng(base + 1000 * seed + fold)
        if scheme == "uniform_pair":                                  # literature-style: circRNA uniform over ALL circRNAs
            out.append(sample_uniform_pairs(np.arange(gd.N["circRNA"]), len(pos), n_dis, known, rng))
        elif scheme == "uniform_test_circ":                           # circRNA uniform over the circRNAs of the evaluated pairs (no held-out artifact)
            out.append(sample_uniform_pairs(fd[part], len(pos), n_dis, known, rng))
        elif scheme == "degree_matched":
            out.append(degree_matched_negatives(pos, deg, n_dis, known, rng))
        else:
            raise ValueError(scheme)
    return out[0], out[1]


def degree_train_negatives(n_dis, pos, deg_edges, exclude_codes, rng, k=5):
    """Strict-protocol training sampler (same circRNA, disease ~ training degree) that excludes ONLY the codes passed in
    (training positives), so it never reads validation or test labels."""
    cnt = np.bincount(deg_edges[:, 1], minlength=n_dis).astype(float)
    p = cnt / cnt.sum()
    c = np.repeat(pos[:, 0], k)
    t = rng.choice(n_dis, size=len(c), p=p)
    for _ in range(50):
        bad = np.isin(codes(c, t, n_dis), exclude_codes)
        if not bad.any():
            break
        t[bad] = rng.choice(n_dis, size=int(bad.sum()), p=p)
    keep = ~np.isin(codes(c, t, n_dis), exclude_codes)
    return np.c_[c[keep], t[keep]].astype(np.int64)


def check_fold_pair(gd, fd, neg_val, neg_test, scheme):
    """Leak assertions of protocol_ladder.yaml for a pair-level fold. Raises AssertionError."""
    n_dis, known = gd.N["disease"], gd.known_codes["disease"]
    E = gd.E["disease"]
    allc = np.sort(codes(E[:, 0], E[:, 1], n_dis))
    cd = {p: codes(fd[f"disease_{p}"][:, 0], fd[f"disease_{p}"][:, 1], n_dis) for p in ("train", "val", "test")}
    cat = np.concatenate([cd["train"], cd["val"], cd["test"]])
    assert len(np.unique(cat)) == len(cat) and np.array_equal(np.sort(cat), allc), "positives are not partitioned into train/val/test exactly once"
    assert set(map(int, fd["train"])) == set(map(int, fd["disease_train"][:, 0])), "'train' circRNAs differ from the circRNAs of the training pairs"
    for name, neg, pos in (("neg_val", neg_val, fd["disease_val"]), ("neg_test", neg_test, fd["disease_test"])):
        nc = codes(neg[:, 0], neg[:, 1], n_dis)
        assert len(neg) == len(pos), f"{name}: {len(neg)} negatives for {len(pos)} positives"
        assert not np.isin(nc, known).any(), f"{name}: contains a known positive"
        assert len(np.unique(nc)) == len(nc), f"{name}: duplicate pairs"
        if scheme == "degree_matched":
            assert np.array_equal(np.sort(neg[:, 0]), np.sort(pos[:, 0])), f"{name}: circRNAs differ from the positives'"
        if scheme == "uniform_test_circ":
            assert set(map(int, neg[:, 0])) <= set(map(int, pos[:, 0])), f"{name}: circRNA outside the evaluated set"
    return True


def label_matrix(pairs, n_c, n_d):
    Y = np.zeros((n_c, n_d))
    Y[pairs[:, 0], pairs[:, 1]] = 1.0
    return Y


def check_masked_matrix(gd, fd, Y_used):
    """The label matrix feeding any kernel/feature must equal the full matrix with validation and test pairs zeroed."""
    Yfull = label_matrix(gd.E["disease"], gd.N["circRNA"], gd.N["disease"])
    for part in ("val", "test"):
        Yfull[fd[f"disease_{part}"][:, 0], fd[f"disease_{part}"][:, 1]] = 0.0
    assert np.array_equal(Yfull, Y_used), "feature/kernel matrix contains validation or test labels"
    return True


def _gip(M):
    n2 = (M ** 2).sum(1)
    gamma = 1.0 / max(n2.mean(), 1e-12)                              # normalised bandwidth
    return np.exp(-gamma * np.maximum(n2[:, None] + n2[None, :] - 2 * M @ M.T, 0.0))


def kernel_propagation(Y):
    """Baseline: GIP kernels from the label matrix Y (training pairs only); score = labels of similar circRNAs
    + the circRNA's own labels on similar diseases (row-normalised kernels, diagonal removed)."""
    Kc, Kd = _gip(Y), _gip(Y.T)
    np.fill_diagonal(Kc, 0.0); np.fill_diagonal(Kd, 0.0)
    rn = lambda K: K / np.maximum(K.sum(1, keepdims=True), 1e-12)
    return rn(Kc) @ Y + Y @ rn(Kd).T


def kernel_scores(Y):
    """The three kernel-propagation baselines, all from the label matrix Y built from TRAINING pairs only:
    kernel_circ = labels of similar circRNAs, kernel_dis = the circRNA's own labels on similar diseases, kernel_both = their sum."""
    Kc, Kd = _gip(Y), _gip(Y.T)
    np.fill_diagonal(Kc, 0.0); np.fill_diagonal(Kd, 0.0)
    rn = lambda K: K / np.maximum(K.sum(1, keepdims=True), 1e-12)
    circ, dis = rn(Kc) @ Y, Y @ rn(Kd).T
    return {"kernel_circ": circ, "kernel_dis": dis, "kernel_both": circ + dis}
