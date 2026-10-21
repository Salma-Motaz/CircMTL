# 10_protocol_ladder - Cell 70: MLP decoder extension, figures and tables (read-only): DistMult against MLP-cat and MLP-catprod under S0 (both negative schemes), A, C and C2, one figure per model (M1, M3).
# DistMult = the frozen run of each protocol; MLP = the folders of Cell 61. Needs Cell 65.
assert "run_report" in globals(), "run Cell 65 first"
REF70 = {"S0": OLD, "A": OUT / "pred_A", "C": OUT / "pred_C", "C2": OUT / "pred_C2"}
PROT70 = (("S0", "strict", None, ("degree_matched", "random")), ("A", "pair", "A", ("degree_matched",)), ("C", "pair", "C", ("uniform_pair",)), ("C2", "pair", "C2", ("uniform_test_circ",)))
for protocol, kind, ptag, schemes in PROT70:
    srcs = {"DistMult": REF70[protocol]}
    for lab, dec in (("MLP-cat", "cat"), ("MLP-catprod", "catprod")):
        f = ROUND1 / f"pred_{protocol}_mlp{dec}"
        if (f / "DONE.txt").exists(): srcs[lab] = f
        else: print(f"[{protocol}] {lab} not complete in {f} (left out)")
    if len(srcs) == 1: continue
    for m in ("M1", "M3"):
        P = pd.concat([read_arm(f, m, label=lab) for lab, f in srcs.items()], ignore_index=True)
        for scheme in schemes:
            print()
            run_report(P, list(srcs), tag=f"{protocol}_MLP_{m}", scheme=scheme, kind=kind, ptag=ptag)
