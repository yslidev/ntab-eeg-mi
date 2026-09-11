"""End-to-end check of the shipped artefact.

Runs predict.py's own code path over raw EDF files from the 12 HOLDOUT
subjects. Those subjects appear nowhere else: not in model selection, not in
any reported cross-validation fold, not in the shipped model's training set.
This is the number to believe about deployment.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import data as D, config as CFG
from predict import load_model, predict_file
import evaluation as E

model = load_model()
print(f"model: trained on {model['n_train_epochs']} epochs from "
      f"{model['n_train_subjects']} subjects, paradigm={model['paradigm']}")
print(f"holdout subjects: {CFG.HOLDOUT_SUBJECTS.tolist()}\n")

rows = []
for par, runs in [("imagined", D.RUNS_IMAG_LR), ("executed", D.RUNS_EXEC_LR)]:
    for s in CFG.HOLDOUT_SUBJECTS:
        for r in runs:
            f = D.CACHE / f"S{int(s):03d}" / f"S{int(s):03d}R{r:02d}.edf"
            if not f.is_file():
                continue
            try:
                pred, proba, onsets, y = predict_file(f, model)
            except SystemExit as e:
                print(f"  {f.name}: {e}"); continue
            rows.append(dict(paradigm=par, subject=int(s), run=r, n=len(pred),
                             correct=int((pred == y).sum()), acc=float((pred == y).mean())))

df = pd.DataFrame(rows)
df.to_csv("results/holdout_predict.csv", index=False)

print(f"{'paradigm':<10}{'n files':>8}{'n trials':>10}{'pooled acc':>12}"
      f"{'95% CI':>18}{'per-subject mean':>18}")
for par, g in df.groupby("paradigm"):
    n, k = int(g.n.sum()), int(g.correct.sum())
    lo, hi = E.binom_ci(k, n)
    agg = g.groupby("subject")[["correct", "n"]].sum()
    bysub = agg.correct / agg.n
    print(f"{par:<10}{len(g):>8}{n:>10}{k/n:>12.4f}"
          f"{f'[{lo:.3f}, {hi:.3f}]':>18}{bysub.mean():>13.4f}"
          f" ± {bysub.std():.3f}")

print("\nper-subject, imagined:")
im = df[df.paradigm == "imagined"].groupby("subject").agg(
    n=("n", "sum"), correct=("correct", "sum"))
im["acc"] = im.correct / im.n
im["sig_thresh"] = [E.binom_sig_threshold(int(v)) for v in im.n]
im["beats_chance_p05"] = im.acc > im.sig_thresh
print(im.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"\n{int(im.beats_chance_p05.sum())}/{len(im)} holdout subjects individually "
      f"beat chance at p<0.05 (one-sided binomial)")
