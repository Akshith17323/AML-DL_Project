"""Write a neuron-level TikZ drawing of the final Phase 2 models.

(a) the final MLP chosen by TPE on all 920 patients (20 -> 32 -> 16 -> 1), every
unit and every weight drawn; (b) the final cross-fitted BN-MLP stack.
Hyperparameters are read from results/phase2/main/metrics/final_hyperparameters.json.

Usage (from the repository root):
    python reports/phase2/ieee/scripts/make_architecture_figure.py \
        results/phase2/main/metrics/final_hyperparameters.json \
        reports/phase2/ieee/figures/final_architecture.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

INPUTS = (["age", "trestbps", "chol", "thalach", "oldpeak"]          # continuous, standardised
          + ["sex", "fbs", "exang"]                                    # binary
          + ["ca", "slope"]                                            # ordinal, standardised
          + ["cp=1", "cp=2", "cp=3", "cp=4", "restecg=0", "restecg=1", "restecg=2",
             "thal=3", "thal=6", "thal=7"])                            # one-hot
GROUPS = [(0, 4, "continuous", "cA"), (5, 7, "binary", "cB"), (8, 9, "ordinal", "cC"), (10, 19, "one-hot", "cD")]
DROP1 = {2, 6, 11, 15, 19, 23, 27, 30, 9}       # an illustrative dropout mask, ~28% of 32
DROP2 = {3, 8, 13, 5}                            # ~25% of 16


def ys(n: int, step: float) -> list[float]:
    return [((n - 1) / 2 - i) * step for i in range(n)]


def mlp_panel(hp: dict) -> list[str]:
    sizes = [int(v) for v in hp["hidden_layout"].split("-")]
    assert sizes == [32, 16], sizes
    X = [2.15, 4.9, 7.45, 9.35]
    Y = [ys(20, 0.40), ys(32, 0.25), ys(16, 0.50), [0.0]]
    R = [0.105, 0.085, 0.12, 0.2]
    drops = [set(), DROP1, DROP2, set()]
    out = []
    # weights: every connection between active units
    for l, colour in ((0, "cW1"), (1, "cW2"), (2, "cW3")):
        for i, y0 in enumerate(Y[l]):
            if i in drops[l]:
                continue
            for j, y1 in enumerate(Y[l + 1]):
                if j in drops[l + 1]:
                    continue
                out.append(f"\\draw[{colour}, line width=0.12pt, opacity=0.45] ({X[l]:.2f},{y0:.3f}) -- ({X[l+1]:.2f},{y1:.3f});")
    # input units, coloured by feature group, with labels
    for a, b, name, col in GROUPS:
        for i in range(a, b + 1):
            out.append(f"\\node[unit, fill={col}, minimum size={2*R[0]:.2f}cm] at ({X[0]},{Y[0][i]:.3f}) {{}};")
            out.append(f"\\node[anchor=east, font=\\fontsize{{5.6}}{{6}}\\selectfont\\ttfamily] at ({X[0]-0.16:.2f},{Y[0][i]:.3f}) {{{INPUTS[i]}}};")
        top, bot = Y[0][a] + 0.17, Y[0][b] - 0.17
        out.append(f"\\draw[{col}!80!black, line width=0.8pt] (0.42,{top:.3f}) -- (0.42,{bot:.3f});")
        out.append(f"\\node[rotate=90, font=\\fontsize{{5.6}}{{6}}\\selectfont\\itshape, text={col}!60!black] at (0.27,{(top+bot)/2:.3f}) {{{name}}};")
    for l, fill in ((1, "cH"), (2, "cH")):
        for i, y in enumerate(Y[l]):
            if i in drops[l]:
                out.append(f"\\node[unit, dashed, draw=black!35, fill=white, minimum size={2*R[l]:.2f}cm] at ({X[l]},{y:.3f}) {{}};")
                out.append(f"\\draw[black!45, line width=0.35pt] ({X[l]-R[l]*0.7:.3f},{y-R[l]*0.7:.3f}) -- ({X[l]+R[l]*0.7:.3f},{y+R[l]*0.7:.3f}) ({X[l]-R[l]*0.7:.3f},{y+R[l]*0.7:.3f}) -- ({X[l]+R[l]*0.7:.3f},{y-R[l]*0.7:.3f});")
            else:
                out.append(f"\\node[unit, fill={fill}, minimum size={2*R[l]:.2f}cm] at ({X[l]},{y:.3f}) {{}};")
    out.append(f"\\node[unit, fill=cO, minimum size={2*R[3]:.2f}cm, font=\\tiny\\bfseries, text=white] (o) at ({X[3]},0) {{$\\sigma$}};")
    out.append(f"\\draw[-{{Latex[length=1.6mm]}}, thick] (o.east) -- ++(0.55,0) node[right, font=\\scriptsize, align=left] {{$\\hat p$\\\\[-1pt]{{\\tiny P(disease)}}}};")
    # layer headers and parameter counts
    top = 4.55
    heads = [("Input layer", "20 features"), ("Hidden layer 1", "32 ReLU + dropout"),
             ("Hidden layer 2", "16 ReLU + dropout"), ("Output", "1 sigmoid")]
    params = ["", r"$20{\times}32{+}32 = 672$", r"$32{\times}16{+}16 = 528$", r"$16{+}1 = 17$"]
    for x, (h, s), p in zip(X, heads, params):
        out.append(f"\\node[font=\\scriptsize\\bfseries, align=center] at ({x},{top}) {{{h}\\\\[-1pt]\\mdseries\\tiny {s}}};")
        if p:
            out.append(f"\\node[font=\\tiny, text=black!70] at ({x - 1.3:.2f},-4.45) {{{p}}};")
    out.append(r"\node[font=\tiny, text=black!70] at (5.4,-4.82) {weights + biases: 672 + 528 + 17 = \textbf{1{,}217 trainable parameters}};")
    out.append(r"\node[font=\tiny, align=left, anchor=north west, text=black!75] at (8.12,-2.2) "
               r"{$\mathbf h^{(l)}=\mathrm{Dropout}_p\big(\mathrm{ReLU}(\mathbf W^{(l)}\mathbf h^{(l-1)}+\mathbf b^{(l)})\big)$\\"
               r"$\hat p=\sigma(\mathbf w^{\top}\mathbf h^{(2)}+b)$\\[2pt]"
               f"dropout $p={hp['dropout']:.2f}$, Adam lr ${hp['learning_rate']*1e4:.1f}{{\\times}}10^{{-4}}$\\\\"
               f"weight decay ${hp['weight_decay']*1e4:.1f}{{\\times}}10^{{-4}}$, batch {hp['batch_size']}\\\\"
               r"loss: binary cross-entropy\\early stopping, patience 15};")
    out.append(r"\node[unit, dashed, draw=black!35, fill=white, minimum size=0.17cm] at (2.35,-4.1) {};")
    out.append(r"\draw[black!45, line width=0.35pt] (2.29,-4.16) -- (2.41,-4.04) (2.29,-4.04) -- (2.41,-4.16);")
    out.append(r"\node[font=\tiny, text=black!60, anchor=west] at (2.5,-4.1) {= unit dropped in one training step (example dropout mask); all units are active at prediction};")
    out.append(r"\node[font=\small\bfseries] at (0.3,4.55) {(a)};")
    return out


def stack_panel(hp: dict) -> list[str]:
    ox = 12.2
    out = [r"\node[font=\small\bfseries] at (%.2f,4.55) {(b)};" % (ox - 0.2),
           r"\node[font=\scriptsize\bfseries, align=center] at (%.2f,4.55) {Final BN--MLP stack\\[-1pt]\mdseries\tiny ML + DL integration};" % (ox + 2.95)]
    out.append(r"\node[draw, rounded corners=2pt, fill=cB, font=\tiny, align=center, minimum width=1.05cm, minimum height=0.8cm] (in) at (%.2f,0) {13 raw\\predictors};" % (ox + 0.35))
    # MLP branch: one hidden layer of 16 units
    hx, top_y = ox + 1.75, 2.15
    hy = ys(16, 0.17)
    out.append(r"\node[draw=black!40, rounded corners=3pt, fill=black!3, minimum width=2.75cm, minimum height=3.3cm] (mbox) at (%.2f,%.2f) {};" % (hx + 0.4, top_y))
    out.append(r"\node[font=\tiny\bfseries, anchor=south] at (%.2f,%.2f) {MLP: 20 $\to$ 16 ReLU $\to$ 1};" % (hx + 0.4, top_y + 1.68))
    ix = hx - 0.55
    iy = ys(8, 0.33)
    for y0 in iy:
        for y1 in hy:
            out.append(f"\\draw[cW1, line width=0.1pt, opacity=0.45] ({ix:.2f},{top_y+y0:.3f}) -- ({hx+0.35:.2f},{top_y+y1:.3f});")
    for y1 in hy:
        out.append(f"\\draw[cW3, line width=0.15pt, opacity=0.6] ({hx+0.35:.2f},{top_y+y1:.3f}) -- ({hx+1.25:.2f},{top_y:.3f});")
    for y0 in iy:
        out.append(f"\\node[unit, fill=cD, minimum size=0.15cm] at ({ix:.2f},{top_y+y0:.3f}) {{}};")
    for y1 in hy:
        out.append(f"\\node[unit, fill=cH, minimum size=0.13cm] at ({hx+0.35:.2f},{top_y+y1:.3f}) {{}};")
    out.append(r"\node[unit, fill=cO, minimum size=0.28cm] (pm) at (%.2f,%.2f) {};" % (hx + 1.25, top_y))
    out.append(r"\node[font=\tiny, text=black!70, align=center] at (%.2f,%.2f) {8 of 20 inputs drawn $\cdot$ dropout %.2f};" % (hx + 0.4, top_y - 1.45, hp["mlp_dropout"]))
    # BN branch: small expert DAG
    by = -2.35
    out.append(r"\node[draw=black!40, rounded corners=3pt, fill=black!3, minimum width=2.45cm, minimum height=3.1cm] (bbox) at (%.2f,%.2f) {};" % (hx + 0.4, by))
    out.append(r"\node[font=\tiny\bfseries, anchor=north] at (%.2f,%.2f) {Bayesian network (expert DAG)};" % (hx + 0.4, by - 1.6))
    bn = {"age": (-0.55, 1.05), "sex": (0.15, 1.05), "chol": (0.85, 1.05), "D": (0.15, 0.2),
          "cp": (-0.6, -0.65), "ca": (0.0, -0.75), "thal": (0.55, -0.75), "slope": (1.05, -0.55)}
    for k, (dx, dy) in bn.items():
        fill = "cO" if k == "D" else "cA!70"
        size = "0.36" if k == "D" else "0.3"
        txt = r"\textcolor{white}{\textbf{D}}" if k == "D" else k
        out.append(f"\\node[unit, fill={fill}, minimum size={size}cm, font=\\fontsize{{4.5}}{{5}}\\selectfont] (bn{k}) at ({hx+0.25+dx:.2f},{by+dy:.2f}) {{{txt}}};")
    for a, b in (("age", "D"), ("sex", "D"), ("chol", "D"), ("D", "cp"), ("D", "ca"), ("D", "thal"), ("D", "slope")):
        out.append(f"\\draw[-{{Latex[length=0.9mm]}}, line width=0.4pt, black!70] (bn{a}) -- (bn{b});")
    out.append(r"\node[font=\fontsize{4.5}{5}\selectfont, text=black!60, align=center] at (%.2f,%.2f) {14 nodes, 16 edges\\CPTs, pseudo-count %.1f};" % (hx + 0.35, by - 1.25, hp["bn_pseudo_count"]))
    out.append(r"\node[unit, fill=cO, minimum size=0.28cm] (pb) at (%.2f,%.2f) {};" % (hx + 1.25, by + 0.2))
    out.append(r"\draw[cO, line width=0.5pt] (bnD) -- (pb);")
    # inputs to both branches
    out.append(r"\draw[-{Latex[length=1.2mm]}, thick, black!70] (in.north) |- (mbox.west);")
    out.append(r"\draw[-{Latex[length=1.2mm]}, thick, black!70] (in.south) |- (bbox.west);")
    # meta perceptron
    mx = ox + 4.55
    out.append(r"\node[unit, fill=cM, minimum size=0.75cm, font=\scriptsize, text=white] (meta) at (%.2f,0) {$\Sigma\,\sigma$};" % mx)
    out.append(r"\draw[-{Latex[length=1.3mm]}, line width=0.9pt, cW3] (pm) to[out=0,in=125] node[pos=0.55, above right=-1pt, font=\tiny, text=black] {$\beta_2$} (meta);")
    out.append(r"\draw[-{Latex[length=1.3mm]}, line width=0.9pt, cW1] (pb) to[out=0,in=235] node[pos=0.55, below right=-1pt, font=\tiny, text=black] {$\beta_1$} (meta);")
    out.append(r"\node[font=\tiny] at (%.2f,%.2f) {$p_{\mathrm{MLP}}$};" % (hx + 1.25, top_y + 0.33))
    out.append(r"\node[font=\tiny] at (%.2f,%.2f) {$p_{\mathrm{BN}}$};" % (hx + 1.25, by + 0.53))
    out.append(r"\draw[-{Latex[length=1.6mm]}, thick] (meta.east) -- ++(0.45,0) node[right, font=\scriptsize] {$\hat p$};")
    out.append(r"\node[font=\tiny, align=center, text=black!75, anchor=south] at (%.2f,2.55) {\textbf{logistic combiner}\\$\hat p=\sigma(\beta_0+\beta_1p_{\mathrm{BN}}+\beta_2p_{\mathrm{MLP}})$\\L2-regularised, $C=%.2f$\\fit only on out-of-fold\\$p_{\mathrm{BN}}, p_{\mathrm{MLP}}$ ($K=3$)};" % (mx + 0.35, hp["meta_C"]))
    return out


def main(hp_path: str, out_path: str) -> None:
    hp = json.loads(Path(hp_path).read_text())
    lines = [r"% generated by scripts/make_architecture_figure.py -- do not edit by hand",
             r"\begin{tikzpicture}[x=1cm, y=1cm, unit/.style={circle, draw=black!65, line width=0.3pt, inner sep=0pt}]",
             r"\definecolor{cA}{HTML}{9EC9E2}\definecolor{cB}{HTML}{D5D9E0}\definecolor{cC}{HTML}{BFE3D3}\definecolor{cD}{HTML}{F6D7A7}",
             r"\definecolor{cH}{HTML}{E8A0AC}\definecolor{cO}{HTML}{A3172F}\definecolor{cM}{HTML}{1B2130}",
             r"\definecolor{cW1}{HTML}{0173B2}\definecolor{cW2}{HTML}{8C96A8}\definecolor{cW3}{HTML}{A3172F}"]
    lines += mlp_panel(hp["MLP"])
    lines += [r"\draw[black!25, line width=0.5pt] (11.75,4.8) -- (11.75,-4.95);"]
    lines += stack_panel(hp["BN-MLP Stack"])
    lines.append(r"\end{tikzpicture}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:3])
