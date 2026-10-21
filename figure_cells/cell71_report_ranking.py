# 10_protocol_ladder - Cell 71: per-circRNA ranking of the true disease among all diseases on S0 (read-only): cumulative hit rate against k for the popularity ranker, a random list and the arms
# DM-a, DM-b, MLP-b and MLPx-b (models M1 and M3), from the full score matrices; filtered rank (the circRNA's other known diseases removed). Figure + table + csv in OUT. Needs Cell 65.
assert "run_report" in globals(), "run Cell 65 first"
ARMS71 = {"DM-a": ROUND1 / "pred_S0rank", "DM-b": ROUND1 / "pred_S0_DMb", "MLP-b": ROUND1 / "pred_S0_MLPb_cat", "MLPx-b": ROUND1 / "pred_S0_MLPb_catprod"}
ARMS71 = {lab: f for lab, f in ARMS71.items() if (f / "DONE.txt").exists()}
K71, SEEDS71, n_dis71 = np.arange(1, 51), range(3), gd.N["disease"]
K_known71 = np.zeros((gd.N["circRNA"], n_dis71), bool); K_known71[gd.E["disease"][:, 0], gd.E["disease"][:, 1]] = True

def filtered_ranks(folder, m, s, popularity=False):
    out, ncand = [], []
    for k in range(5):
        z = np.load(folder / f"full_{m}_seed{s}_fold{k}.npz")
        rows = pd.Series(np.arange(len(z["test_c"])), index=z["test_c"]).loc[z["pos_c"]].to_numpy()
        Sg = np.tile(z["train_degree"].astype(float), (len(rows), 1)) if popularity else z["scores"][rows]
        st = Sg[np.arange(len(rows)), z["pos_t"]]; other = K_known71[z["pos_c"]].copy(); other[np.arange(len(rows)), z["pos_t"]] = False
        out.append(1 + ((Sg > st[:, None]) & ~other).sum(1) + 0.5 * (((Sg == st[:, None]) & ~other).sum(1) - 1)); ncand.append(n_dis71 - other.sum(1))
    return np.concatenate(out), np.concatenate(ncand)

def curves(rank_by_seed, ncand_by_seed):
    hit = np.mean([[np.mean(r <= k) for k in K71] for r in rank_by_seed], axis=0)
    mrr = np.mean([np.mean(1.0 / r) for r in rank_by_seed]); med = np.mean([np.median(r) for r in rank_by_seed]); return hit, mrr, med
first = next(iter(ARMS71.values()))
pop = [filtered_ranks(first, "M1", s, popularity=True) for s in SEEDS71]
rand_hit = np.mean([[np.mean(np.minimum(k, nc) / nc) for k in K71] for _, nc in pop], axis=0)
rand_mrr = np.mean([np.mean([np.sum(1.0 / np.arange(1, n + 1)) / n for n in np.unique(nc)]) for _, nc in pop])
pop_hit, pop_mrr, pop_med = curves([r for r, _ in pop], [n for _, n in pop])
rows, fig_data = [{"ranker": "popularity (training degree)", "MRR": pop_mrr, "median rank": pop_med, **{f"hit@{k}": pop_hit[k - 1] for k in (1, 5, 10, 20, 50)}},
                  {"ranker": "random list (expected)", "MRR": rand_mrr, "median rank": np.nan, **{f"hit@{k}": rand_hit[k - 1] for k in (1, 5, 10, 20, 50)}}], {}
for m in ("M1", "M3"):
    for lab, f in ARMS71.items():
        rk = [filtered_ranks(f, m, s) for s in SEEDS71]; hit, mrr, med = curves([r for r, _ in rk], [n for _, n in rk]); fig_data[(m, lab)] = hit
        rows.append({"ranker": f"{lab} {m}", "MRR": mrr, "median rank": med, **{f"hit@{k}": hit[k - 1] for k in (1, 5, 10, 20, 50)}})
T71 = pd.DataFrame(rows).set_index("ranker"); print(f"per-circRNA ranking on S0 (mean over {len(list(SEEDS71))} seeds; filtered rank among {n_dis71} diseases)"); print(T71.round(4).to_string())
T71.to_csv(OUT / "report_S0_ranking.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, m in zip(axes, ("M1", "M3")):
    for lab in ARMS71: ax.plot(K71, fig_data[(m, lab)], label=f"{lab} ({T71.loc[f'{lab} {m}', 'hit@10']:.3f} at k=10)")
    ax.plot(K71, pop_hit, color="black", ls="--", label=f"popularity ({pop_hit[9]:.3f})"); ax.plot(K71, rand_hit, color="grey", ls=":", label=f"random list ({rand_hit[9]:.3f})")
    ax.set(xlabel="k (top-k diseases of the circRNA)", title=f"{m}: true disease within the top k [S0]"); ax.legend(fontsize=7, loc="lower right")
axes[0].set_ylabel("fraction of test positives"); plt.tight_layout(); plt.savefig(OUT / "report_S0_ranking.png", dpi=200); plt.show()
