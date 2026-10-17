"""Strict check of configs/protocol_ladder.yaml (read-only).
Usage (project root):  python scripts/check_registration.py [path-to-yaml]
Reports ERRORS (must fix) and WARNINGS (pending edits). Unlike yaml.safe_load, it fails on duplicate keys,
which plain loading silently resolves by keeping the LAST value (and so silently drops the earlier line).
"""
import re, sys
from pathlib import Path
import yaml
from yaml.constructor import ConstructorError

CFG = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/protocol_ladder.yaml")

REQUIRED_TOP = ["purpose", "main_result", "runs", "models", "baselines", "seeds", "folds", "unchanged_across_runs", "model_selection", "threshold",
                "literature_alignment", "pair_level_split", "negatives", "leak_assertions", "metrics", "comparisons", "reporting_rules",
                "leak_audit_control", "S0_threshold_note", "shuffled_auxiliary_control", "shuffled_auxiliary_control_outcome",
                "baseline_code_audit", "reported_baseline_numbers", "encoder_dev_sweep", "run_log"]
REQUIRED_RUN_LOG = ["S0", "frozen_pair", "regression_test", "B", "A", "C", "C2", "leak_audit", "shuffled_aux_A", "shuffled_aux_C", "shuffled_aux_C2",
                    "aux_channel_control", "simple_learner_reference", "trivial_sequence_reference", "gd_check", "notebook_state_repair"]

class Strict(yaml.SafeLoader):
    pass
def _construct(loader, node, deep=False):
    seen, dups = {}, []
    for k_node, _ in node.value:
        k = loader.construct_object(k_node, deep=deep)
        if k in seen:
            dups.append(f"'{k}' (lines {seen[k]} and {k_node.start_mark.line + 1})")
        else:
            seen[k] = k_node.start_mark.line + 1
    if dups:
        raise ConstructorError(None, None, "duplicate keys: " + "; ".join(dups), node.start_mark)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)
Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct)

def main():
    if not CFG.exists():
        sys.exit(f"not found: {CFG.resolve()} (run from the project root)")
    errors, warns = [], []
    try:
        d = yaml.load(CFG.read_text(encoding="utf-8"), Loader=Strict)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
    for k in REQUIRED_TOP:
        if k not in d:
            errors.append(f"missing top-level block: {k}")
    rl = d.get("run_log", {})
    for k in REQUIRED_RUN_LOG:
        if k not in rl:
            errors.append(f"missing run_log entry: {k}")
    for r in ("A", "C", "C2"):
        v = str(rl.get(f"shuffled_aux_{r}", ""))
        if v and not re.search(r"\d[\d,]* s\)", v):
            warns.append(f"run_log.shuffled_aux_{r} has no run time (add it from the run's last printed line, or leave it as is)")
    if "not yet audited" in str(d.get("baseline_code_audit", {}).get("CLCDA", "")):
        warns.append("baseline_code_audit.CLCDA still says 'not yet audited' (replace it with the audit finding)")
    if "C2" not in str(d.get("S0_threshold_note", "")):
        warns.append("S0_threshold_note does not mention run C2 (it should say: runs B, A, C and C2 use the validation-chosen threshold)")
    if "1000 iterations" not in str(rl.get("simple_learner_reference", "")):
        warns.append("run_log.simple_learner_reference is the older version (without the 1000-iteration rerun)")
    print(f"{CFG}: strict parse OK | {len(d)} top-level blocks | run_log has {len(rl)} entries")
    for e in errors:
        print("ERROR:  ", e)
    for w in warns:
        print("WARNING:", w)
    if not errors and not warns:
        print("all checks passed")
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
