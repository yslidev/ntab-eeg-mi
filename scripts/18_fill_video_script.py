"""Substitute the {{PLACEHOLDER}} tokens in VIDEO_SCRIPT.md from the result CSVs,
so the spoken numbers cannot drift from the tables."""
import pathlib, re
import pandas as pd

R = pathlib.Path("results")
v = {}


def pick(df, col, **eq):
    m = pd.Series(True, index=df.index)
    for k, val in eq.items():
        m &= (df[k] == val) if not isinstance(val, str) or not val.endswith("*") \
            else df[k].str.startswith(val[:-1])
    return df[m][col]


mr = R / "main_regimes.csv"
if mr.is_file():
    d = pd.read_csv(mr)
    d = d[(d.paradigm == "imagined") & (~d.shuffled)]
    g = lambda p: d[d.regime.str.startswith(p)].pooled_acc.iloc[0]
    v["LOSO"] = f"{g('C')*100:.1f}%"
    v["WITHIN_RANDOM"] = f"{g('A')*100:.1f}%"
    v["WITHIN_RUN"] = f"{g('B')*100:.1f}%"
    v["CALIB"] = f"{g('D')*100:.1f}%"

nv = R / "naive_splits.csv"
if nv.is_file():
    d = pd.read_csv(nv)
    naive = d[d.tag.str.startswith("NAIVE")].acc.max()
    v["NAIVE"] = f"{naive*100:.1f}%"

c = R / "controls.csv"
if c.is_file():
    d = pd.read_csv(c)
    def find(block, sub):
        r = d[(d.block == block) & (d.tag.str.contains(sub, regex=False))]
        return f"{r.pooled_acc.iloc[0]*100:.1f}%" if len(r) else "n/a"
    v["PRECUE"] = find("window", "PRE-CUE -2.4")
    v["OCCIP"] = find("lesion", "parieto-occipital")
    v["MOTOR"] = find("lesion", "sensorimotor strip")
    v["CARRY"] = find("carryover", "PREVIOUS")
    v["IDENT_RAW"] = find("identity", "no alignment")
    v["IDENT_ALIGNED"] = find("identity", "after alignment")
    a1 = d[(d.block == "align-scope") & d.tag.str.contains("1st run only")]
    a3 = d[(d.block == "align-scope") & d.tag.str.contains("all 3 runs")]
    if len(a1) and len(a3):
        v["ALIGN_COST"] = f"{(a3.pooled_acc.iloc[0]-a1.pooled_acc.iloc[0])*100:.1f}"

h = R / "holdout_predict.csv"
if h.is_file():
    d = pd.read_csv(h)
    g = d[d.paradigm == "imagined"]
    v["HOLDOUT"] = f"{g.correct.sum()/g.n.sum()*100:.1f}%"

p = pathlib.Path("VIDEO_SCRIPT.md")
s = p.read_text()
missing = set(re.findall(r"\{\{(\w+)\}\}", s)) - set(v)
for k, val in v.items():
    s = s.replace("{{" + k + "}}", val)
p.write_text(s)
print("filled:", ", ".join(f"{k}={val}" for k, val in sorted(v.items())))
if missing:
    print("still unfilled (edit by hand):", sorted(missing))
