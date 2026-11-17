"""Collect the numbers the deck shows from the saved Phase 1 and Phase 2 runs."""
import json, sys
import pandas as pd
from PIL import Image
P2, P1, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
ORDER = ["Bayesian Network","Naive Bayes","Logistic Regression","SVM","Random Forest","Gradient Boosting","MLP","BN-MLP Stack"]
LBL = ["Bayesian network","Naive Bayes","Logistic Regression","SVM","Random Forest","Gradient Boosting","MLP","BN–MLP stack"]
sp = pd.read_csv(f"{P2}/summary_pooled.csv"); sx = pd.read_csv(f"{P2}/summary_cross-cohort.csv")
a = sp[sp.cohort=="ALL"].set_index("model").loc[ORDER]
coh = ["Cleveland","Hungarian","Switzerland","VA"]
loco_mean = [sx[(sx.model==m)&(sx.cohort.isin(coh))].roc_auc.mean() for m in ORDER]
p1 = pd.read_csv(f"{P1}/bootstrap_ci.csv")
print(p1.head(3).to_string())
p1a = p1[p1.metric=="roc_auc"].set_index("model").estimate
P1ORDER = ORDER[:6]
ab = pd.read_csv(f"{P2}/ablations.csv")
abl_names = {"full":"MLP full","no_dropout":"− dropout","no_weight_decay":"− weight decay","no_early_stopping":"− early stopping","one_hidden_layer":"one hidden layer",
             "BN":"BN alone","MLP":"MLP alone","simple_average":"simple average","learned_stack":"learned stack"}
shap = pd.read_csv(f"{P2}/mlp_shap_importance.csv").head(6)
cp = pd.read_csv(f"{P2}/cohort_prevalence.csv").set_index("cohort").loc[coh]
grey, red, blue = "5B6475", "A3172F", "0173B2"
table = []
for m, l in zip(ORDER, LBL):
    r = a.loc[m]; table.append([l, f"{r.brier:.3f}", f"{r.log_loss:.3f}", f"{r.ece:.3f}"])
data = {
  "labels": LBL,
  "colors": [grey]*6 + [red, blue],
  "pooled": {"auc": [round(v,3) for v in a.roc_auc], "table": table},
  "loco": {"mean": [round(v,3) for v in loco_mean]},
  "p1": {"labels": LBL[:6], "auc": [round(p1a[m],3) for m in P1ORDER]},
  "cohort": {"labels": ["Cleveland","Hungarian","Switzerland","Long Beach VA"], "prev": [round(v,4) for v in cp.positive_rate]},
  "abl": {"labels": [abl_names[v] for v in ab.variant], "auc": [round(v,3) for v in ab.roc_auc_mean],
          "colors": [red if s=="mlp" else blue for s in ab.suite]},
  "shap": {"labels": list(shap.feature), "vals": [round(v,3) for v in shap.mean_abs_shap]},
}
json.dump(data, open("data.json","w"), indent=1, ensure_ascii=False)
im = Image.open(f"{FIG}/mlp_vs_bn_interaction_comparison.png"); w,h = im.size
im.crop((0, int(h*0.045), w, int(h*0.53))).save("img/surface_top.png")
print(json.dumps(data)[:600])
