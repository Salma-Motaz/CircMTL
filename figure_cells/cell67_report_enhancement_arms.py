# 10_protocol_ladder - Cell 67: enhancement round 1, figures and tables (read-only): arms "depth" and "sim" on S0 (both negative schemes), A and C2; models M1, M2, M3 as in the main figures.
# Arms whose folder is not complete are skipped. Needs Cell 65.
assert "run_report" in globals(), "run Cell 65 first"
PROTOCOLS67 = (("S0", "strict", None, ("degree_matched", "random")), ("A", "pair", "A", ("degree_matched",)), ("C2", "pair", "C2", ("uniform_test_circ",)))
for protocol, kind, ptag, schemes in PROTOCOLS67:
    for arm in ("depth", "sim"):
        folder = ROUND1 / f"pred_{protocol}_{arm}"
        if not (folder / "DONE.txt").exists():
            print(f"\n[{protocol} {arm}] not complete in {folder} (skipped)"); continue
        P = pd.concat([read_arm(folder, m) for m in ("M1", "M2", "M3")], ignore_index=True)
        for scheme in schemes:
            print()
            run_report(P, ["M1", "M2", "M3"], tag=f"{protocol}_{arm}", scheme=scheme, kind=kind, ptag=ptag)
