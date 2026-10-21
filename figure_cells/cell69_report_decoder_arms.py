# 10_protocol_ladder - Cell 69: decoder arms on S0, figures and tables (read-only): DM-b (DistMult, random-disease training negatives), MLP-b (MLP decoder, same negatives) and MLPx-b (MLP with the
# product input), one figure per model (M1, M3) and per test negative scheme. The S0 figure of the main models (DM-a: degree-matched training negatives) is the one of Cell 1. Needs Cell 65.
assert "run_report" in globals(), "run Cell 65 first"
SRC69 = {"DM-b": ROUND1 / "pred_S0_DMb", "MLP-b": ROUND1 / "pred_S0_MLPb_cat", "MLPx-b": ROUND1 / "pred_S0_MLPb_catprod"}
SRC69 = {lab: f for lab, f in SRC69.items() if (f / "DONE.txt").exists()}
print("arms found:", list(SRC69))
for m in ("M1", "M3"):
    for scheme in ("degree_matched", "random"):
        P = pd.concat([read_arm(f, m, label=lab) for lab, f in SRC69.items()], ignore_index=True)
        print()
        run_report(P, list(SRC69), tag=f"S0_decoder_arms_{m}", scheme=scheme, kind="strict")
