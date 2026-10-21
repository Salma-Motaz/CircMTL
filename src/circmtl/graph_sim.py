"""Enhancement round 1, arm 'sim': two static, label-free relations added to the CircMTL graph.

csim: circRNA-circRNA edges (each circRNA to its k nearest circRNAs by RNA-FM cosine similarity)
dsim: disease-disease edges (each disease to its k most similar diseases by MeSH similarity)

Both are undirected (stored once as (a, b) with a < b; added in both directions with ONE relation type each),
come from sequence embeddings and the ontology only, are never decoded, and sit outside the 70/30 split.
Place this file at src/circmtl/graph_sim.py.
"""
from __future__ import annotations

import numpy as np

from circmtl.graph import GraphData, RELS

STATIC = ("csim", "dsim")            # relation names; their relation ids follow the 2 * len(RELS) existing ones
N_EXTRA = len(STATIC)


def knn_edges(S, k: int, min_sim: float | None = None) -> np.ndarray:
    """Undirected k-nearest-neighbour edges from a similarity matrix.

    Self-similarity is excluded; ties are broken by node index (stable sort), so the result is deterministic.
    min_sim: if given, a neighbour is kept only if its similarity is strictly greater than min_sim.
    Returns an (m, 2) int64 array with a < b, each undirected pair once.
    """
    S = np.array(S, dtype=float, copy=True)
    n = S.shape[0]
    assert S.shape == (n, n), "similarity matrix must be square"
    np.fill_diagonal(S, -np.inf)
    nbr = np.argsort(-S, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = nbr.ravel()
    sim = S[rows, cols]
    keep = np.isfinite(sim) if min_sim is None else (np.isfinite(sim) & (sim > min_sim))
    e = np.c_[rows[keep], cols[keep]]
    e = np.unique(np.sort(e, axis=1), axis=0)
    return e.astype(np.int64)


class GraphDataSim(GraphData):
    """A GraphData plus the two static relations. Folds, negatives, features and the 70/30 split are those of the wrapped object."""

    def __init__(self, gd: GraphData, csim, dsim):
        self.__dict__.update(gd.__dict__)            # share every attribute of the real graph; nothing is copied or changed
        self.static = {"csim": np.asarray(csim, dtype=np.int64), "dsim": np.asarray(dsim, dtype=np.int64)}
        assert self.static["csim"].max() < self.N["circRNA"] and self.static["dsim"].max() < self.N["disease"]

    def _with_static(self, rel: dict) -> dict:
        out = dict(rel)
        out.update(self.static)
        return out

    def epoch_graph(self, fd, rng, model: str):
        mp, sup = super().epoch_graph(fd, rng, model)
        return self._with_static(mp), sup            # static edges are message passing only, never supervision

    def inference_graph(self, fd, stage: str, model: str, with_test_aux: bool = True):
        return self._with_static(super().inference_graph(fd, stage, model, with_test_aux))

    def to_edge_tensors(self, rel: dict, device="cpu"):
        import torch
        ei, et = super().to_edge_tensors({r: e for r, e in rel.items() if r not in STATIC}, device)
        src, dst, typ = [], [], []
        for j, r in enumerate(STATIC):
            if r not in rel:
                continue
            g = np.asarray(rel[r], dtype=np.int64) + (self.offset["disease"] if r == "dsim" else 0)
            src += [g[:, 0], g[:, 1]]
            dst += [g[:, 1], g[:, 0]]
            typ.append(np.full(2 * len(g), 2 * len(RELS) + j))
        if not src:
            return ei, et
        ei2 = torch.from_numpy(np.vstack([np.concatenate(src), np.concatenate(dst)])).long().to(device)
        et2 = torch.from_numpy(np.concatenate(typ)).long().to(device)
        return torch.cat([ei, ei2], 1), torch.cat([et, et2])
