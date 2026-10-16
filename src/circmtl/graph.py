"""CircMTL heterogeneous graph: per-fold splits, epoch graphs, inference graphs and negatives.

All graph logic is numpy/pandas; PyTorch is only needed in `to_edge_tensors` and `to_heterodata`.
Built and checked in notebook 07_graph (Cells 2-4). Rules: configs/graph.yaml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

AUX = ("mirna", "rbp")
RELS = ("disease",) + AUX
REL = {"disease": ("circRNA", "associated_with", "disease"),
       "mirna": ("circRNA", "binds", "mirna"),
       "rbp": ("circRNA", "bound_by", "rbp")}
REL_ID = {r: i for i, r in enumerate(RELS)}          # forward relation ids 0-2; reverse ids 3-5
NODE_ORDER = ("circRNA", "disease", "mirna", "rbp")  # order of the global node index


def _codes(c, t, n_t):
    return c.astype(np.int64) * n_t + t.astype(np.int64)


class GraphData:
    """Loads the frozen graph tables (07_graph Cell 1) and builds fold-specific graphs."""

    def __init__(self, D: Path, G: Path, val_frac: float = 0.2, sup_frac: float = 0.3):
        self.D, self.G = Path(D), Path(G)
        self.val_frac, self.sup_frac = val_frac, sup_frac
        self.nodes_c = pd.read_csv(G / "nodes_circrna.csv", keep_default_na=False)
        nodes_d = pd.read_csv(G / "nodes_disease.csv")
        self.X_raw = np.load(G / "x_circrna.npy")
        self.E = {r: pd.read_csv(G / f"edges_circrna_{r}.csv")[["src", "dst"]].to_numpy(np.int64) for r in RELS}
        self.N = {"circRNA": len(self.nodes_c), "disease": len(nodes_d),
                  "mirna": len(pd.read_csv(G / "nodes_mirna_family.csv")),
                  "rbp": len(pd.read_csv(G / "nodes_rbp.csv", keep_default_na=False))}
        self.offset, o = {}, 0
        for t in NODE_ORDER:
            self.offset[t] = o
            o += self.N[t]
        self.n_total = o

        self.folds = pd.read_csv(D / "folds.csv")
        assert self.folds["circbase_id"].tolist() == self.nodes_c["circbase_id"].tolist(), "folds.csv out of order"
        self.negs = pd.read_csv(D / "negatives_cda.csv")
        self.C_ID = dict(zip(self.nodes_c["circbase_id"], self.nodes_c["circ_idx"]))
        self.D_ID = dict(zip(nodes_d["disease"], nodes_d["disease_id"]))
        self.groups = self.nodes_c["host_group"].to_numpy()
        self.KNOWN_CD = set(map(tuple, self.E["disease"]))

        # Auxiliary labels: circRNAs absent from ENCORI are UNKNOWN; any ENCORI edge (any evidence) is never a negative
        lab = pd.read_csv(D / "circrnas.csv", keep_default_na=False, dtype={"mirna_labelled": bool, "rbp_labelled": bool})
        assert lab["circbase_id"].tolist() == self.nodes_c["circbase_id"].tolist()
        self.LABELLED = {"mirna": lab["mirna_labelled"].to_numpy(), "rbp": lab["rbp_labelled"].to_numpy()}
        seed_of = pd.read_csv(G / "mirna_families.csv").set_index("mirna_id")["seed"]
        fam_idx = pd.read_csv(G / "nodes_mirna_family.csv").set_index("seed")["fam_idx"]
        rbp_idx = pd.read_csv(G / "nodes_rbp.csv", keep_default_na=False).set_index("rbp")["rbp_idx"]
        am = pd.read_csv(D / "mirna_edges.csv", usecols=["circbase_id", "mirna_id"])
        ar = pd.read_csv(D / "rbp_edges.csv", keep_default_na=False, usecols=["circbase_id", "rbp"])
        am = pd.DataFrame({"c": am["circbase_id"].map(self.C_ID), "t": am["mirna_id"].map(seed_of).map(fam_idx)}).dropna().astype(int)
        ar = pd.DataFrame({"c": ar["circbase_id"].map(self.C_ID), "t": ar["rbp"].map(rbp_idx)}).dropna().astype(int)
        self.ANY_AUX = {"mirna": set(zip(am["c"], am["t"])), "rbp": set(zip(ar["c"], ar["t"]))}
        for r in AUX:
            assert set(map(tuple, self.E[r])) <= self.ANY_AUX[r], f"{r}: working edge missing from the any-evidence set"
        # fast lookup codes for training negatives (never a known positive / ENCORI edge)
        self.known_codes = {"disease": np.unique(_codes(self.E["disease"][:, 0], self.E["disease"][:, 1], self.N["disease"]))}
        for r, s in (("mirna", am), ("rbp", ar)):
            self.known_codes[r] = np.unique(_codes(s["c"].to_numpy(), s["t"].to_numpy(), self.N[r]))

    # ------------------------------------------------------------------ frozen negatives (same algorithm as 07_graph)
    @staticmethod
    def _rows(edges, mask):
        return edges[mask[edges[:, 0]]]

    def _sample_frozen(self, pos, deg_edges, n_t, known, seed):
        rng = np.random.default_rng(seed)
        cnt = np.bincount(deg_edges[:, 1], minlength=n_t).astype(float)
        p = cnt / cnt.sum()
        c, t = pos[:, 0].copy(), rng.choice(n_t, size=len(pos), p=p)
        for _ in range(100):
            bad = np.array([(a, b) in known for a, b in zip(c, t)], bool)
            if not bad.any():
                break
            t[bad] = rng.choice(n_t, size=int(bad.sum()), p=p)
        keep = ~np.array([(a, b) in known for a, b in zip(c, t)], bool)
        return np.c_[c[keep], t[keep]].astype(np.int64)

    def build_fold(self, seed: int, fold: int) -> dict:
        """All index arrays for one seed/fold. Test circRNAs are absent from training (inductive)."""
        fcol = self.folds[f"fold_seed{seed}"].to_numpy()
        test, rest = np.flatnonzero(fcol == fold), np.flatnonzero(fcol != fold)
        tr_i, va_i = next(GroupShuffleSplit(1, test_size=self.val_frac, random_state=1000 * seed + fold)
                          .split(rest, groups=self.groups[rest]))
        train, val = np.sort(rest[tr_i]), np.sort(rest[va_i])
        n = self.N["circRNA"]
        m = {k: np.isin(np.arange(n), v) for k, v in (("train", train), ("val", val), ("test", test))}
        xt = self.X_raw[train].astype(np.float64)
        fd = {"seed": seed, "fold": fold, "train": train, "val": val, "test": test,
              "mu": xt.mean(0), "sd": xt.std(0) + 1e-8}
        for r in RELS:
            for part in ("train", "val", "test"):
                fd[f"{r}_{part}"] = self._rows(self.E[r], m[part])
        fd["neg_val"] = self._sample_frozen(fd["disease_val"], fd["disease_train"], self.N["disease"],
                                            self.KNOWN_CD, seed=10_000 + 1000 * seed + fold)
        for i, r in enumerate(AUX):
            pool = np.vstack([fd[f"{r}_train"], fd[f"{r}_val"]])
            fd[f"neg_test_{r}"] = self._sample_frozen(fd[f"{r}_test"], pool, self.N[r], self.ANY_AUX[r],
                                                      seed=20_000 + 1000 * seed + 10 * fold + i)
        for scheme in ("degree_matched", "random"):
            ng = self.negs[(self.negs["seed"] == seed) & (self.negs["fold"] == fold) & (self.negs["scheme"] == scheme)]
            fd[f"neg_test_{scheme}"] = np.c_[ng["circbase_id"].map(self.C_ID), ng["disease"].map(self.D_ID)].astype(np.int64)
        return fd

    def load_fold(self, seed: int, fold: int) -> dict:
        """The frozen arrays saved by 07_graph Cell 4."""
        z = np.load(self.G / "folds" / f"seed{seed}_fold{fold}.npz")
        fd = {k: z[k] for k in z.files}
        fd["seed"], fd["fold"] = seed, fold
        return fd

    # ------------------------------------------------------------------ features and graphs
    def features(self, fd) -> np.ndarray:
        return ((self.X_raw.astype(np.float64) - fd["mu"]) / fd["sd"]).astype(np.float32)

    def epoch_graph(self, fd, rng, model: str):
        """Disjoint 70/30 message-passing / supervision split of the training edges (fresh each epoch).
        Pass a dedicated rng: M2 and M3 then see identical graphs."""
        mp, sup = {}, {}
        rels = ("disease",) if model == "M1" else RELS
        for r in rels:
            s = rng.random(len(fd[f"{r}_train"])) < self.sup_frac
            mp[r], sup[r] = fd[f"{r}_train"][~s], fd[f"{r}_train"][s]
        if model != "M1":
            for r in AUX:                                    # validation circRNAs: auxiliary edges are input only
                mp[r] = np.vstack([mp[r], fd[f"{r}_val"]])
        return mp, sup

    def inference_graph(self, fd, stage: str, model: str, with_test_aux: bool = True):
        """Disease edges: training circRNAs only, identical at validation and test.
        Auxiliary edges: train + val circRNAs; at test, plus the test circRNAs' if with_test_aux."""
        rel = {"disease": fd["disease_train"]}
        if model != "M1":
            for r in AUX:
                parts = [fd[f"{r}_train"], fd[f"{r}_val"]]
                if stage == "test" and with_test_aux:
                    parts.append(fd[f"{r}_test"])
                rel[r] = np.vstack(parts)
        return rel

    # ------------------------------------------------------------------ training negatives
    def train_negatives(self, r: str, pos, deg_edges, rng, k: int = 5):
        """k negatives per supervision edge: same circRNA, partner ~ training degree, never a known positive
        (disease) or an ENCORI edge of any evidence (auxiliary). Supervision circRNAs are labelled by construction."""
        n_t = self.N[r]
        cnt = np.bincount(deg_edges[:, 1], minlength=n_t).astype(float)
        p = cnt / cnt.sum()
        c = np.repeat(pos[:, 0], k)
        t = rng.choice(n_t, size=len(c), p=p)
        for _ in range(50):
            bad = np.isin(_codes(c, t, n_t), self.known_codes[r])
            if not bad.any():
                break
            t[bad] = rng.choice(n_t, size=int(bad.sum()), p=p)
        keep = ~np.isin(_codes(c, t, n_t), self.known_codes[r])
        return np.c_[c[keep], t[keep]].astype(np.int64)

    # ------------------------------------------------------------------ tensors
    def global_pairs(self, r: str, pairs):
        """(circRNA idx, partner idx) -> global node indices."""
        out = pairs.copy()
        out[:, 1] = out[:, 1] + self.offset[REL[r][2]]
        return out

    def to_edge_tensors(self, rel: dict, device="cpu"):
        """Homogeneous edge_index / edge_type for RGCN. Reverse edges are built ONLY from `rel`."""
        import torch
        src, dst, typ = [], [], []
        for r, e in rel.items():
            g = self.global_pairs(r, e)
            src += [g[:, 0], g[:, 1]]
            dst += [g[:, 1], g[:, 0]]
            typ += [np.full(len(g), REL_ID[r]), np.full(len(g), REL_ID[r] + len(RELS))]
        ei = torch.from_numpy(np.vstack([np.concatenate(src), np.concatenate(dst)])).long().to(device)
        et = torch.from_numpy(np.concatenate(typ)).long().to(device)
        return ei, et

    def to_heterodata(self, x, rel: dict):
        """PyTorch Geometric HeteroData (for HGT); reverse edges only from `rel`."""
        import torch
        from torch_geometric.data import HeteroData
        h = HeteroData()
        h["circRNA"].x = torch.from_numpy(x)
        for r, e in rel.items():
            s, name, t = REL[r]
            h[t].num_nodes = self.N[t]
            ei = torch.from_numpy(np.ascontiguousarray(e.T)).long()
            h[s, name, t].edge_index = ei
            h[t, "rev_" + name, s].edge_index = ei.flip(0)
        return h
