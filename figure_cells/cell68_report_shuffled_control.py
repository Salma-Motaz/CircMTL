# 10_protocol_ladder - Cell 68: shuffled-auxiliary control, figures and tables (read-only): M3 against M3_shuf (partner edges reassigned among circRNAs) on A, C and C2. A control, not a result.
# Needs Cell 65.
assert "run_report" in globals(), "run Cell 65 first"
for run, scheme in (("A", "degree_matched"), ("C", "uniform_pair"), ("C2", "uniform_test_circ")):
    ctl = OUT / f"pred_{run}_M3shuf"
    if not (ctl / "DONE.txt").exists():
        print(f"\n[{run}] control not complete in {ctl} (skipped)"); continue
    P = pd.concat([read_arm(OUT / f"pred_{run}", "M3"), read_arm(ctl, "M3_shuf", pattern="pred_M3shuf_seed*_fold*.csv")], ignore_index=True)
    print()
    run_report(P, ["M3", "M3_shuf"], tag=f"{run}_shuffled_control", scheme=scheme, kind="pair", ptag=run)
