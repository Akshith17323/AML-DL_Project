"""Fail if headline numbers quoted in main.tex disagree with the saved run.

Run after make_ieee_tables.py (which writes tables/key_numbers.json):
    python reports/phase2/ieee/scripts/audit_report_numbers.py reports/phase2/ieee
"""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
tex = (root / "main.tex").read_text()
keys = json.loads((root / "tables" / "key_numbers.json").read_text())
pooled, held = keys["pooled_auc"], keys["mean_heldout_auc"]
checks = {
    "pooled AUC range": f"{min(pooled.values()):.3f}--{max(pooled.values()):.3f}",
    "MLP pooled AUC": f"{pooled['MLP']:.3f}",
    "stack pooled AUC": f"{pooled['BN-MLP Stack']:.3f}",
    "held-out AUC range": f"{min(held.values()):.3f}--{max(held.values()):.3f}",
    "NB held-out AUC": f"NB {held['Naive Bayes']:.3f}",
    "MLP held-out AUC": f"MLP ({held['MLP']:.3f})",
    "pooled trial count": f"{keys['n_trials_total_pooled']} pooled",
    "LOCO trial count": f"{keys['n_trials_total_cross']} LOCO",
}
drops = keys["loco_minus_pooled_auc"].values()
checks["LOCO drop range"] = f"{-max(drops):.3f}--{-min(drops):.3f}"
failed = [f"{name}: expected '{text}'" for name, text in checks.items() if text not in tex]
for line in failed:
    print("MISMATCH", line)
print("audit:", "FAILED" if failed else f"all {len(checks)} checks passed")
sys.exit(1 if failed else 0)
