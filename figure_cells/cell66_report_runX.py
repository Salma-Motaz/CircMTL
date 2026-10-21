# 10_protocol_ladder - Cell 66: run X, figures and tables (read-only): primary (degree-matched) and secondary (uniform) negatives; the 2-layer models M1, M2, M3 and the 3-layer models M1d, M2d, M3d.
# Baselines: disease popularity from the dataset_v1 degrees; circRNA degree is 0 for new circRNAs. Needs Cell 65.
assert "run_report" in globals(), "run Cell 65 first"
XD = ROUND1 / "pred_X"
assert (XD / "DONE.txt").exists(), "run X is not complete"
for scheme, name in (("degree_matched", "X_primary"), ("uniform_pair", "X_secondary")):
    for models, suffix in ((["M1", "M2", "M3"], ""), (["M1d", "M2d", "M3d"], "_3layer")):
        P = pd.concat([read_arm(XD, m) for m in models], ignore_index=True)
        print()
        run_report(P, models, tag=name + suffix, scheme=scheme, kind="external")
