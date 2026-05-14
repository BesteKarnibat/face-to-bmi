"""Generate the supporting figures embedded in DATA_PIPELINE.md.

Run from the repo root:
    python scripts/generate_eda_figures.py

Outputs go to docs/figures/. Re-run any time the underlying CSVs change.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(DATA / "raw" / "data.csv").rename(columns={"Unnamed: 0": "row_id"})
clean = pd.read_csv(DATA / "processed" / "clean_data.csv").rename(columns={"Unnamed: 0": "row_id"})
audit = pd.read_csv(DATA / "processed" / "audit_log.csv")
standardize = pd.read_csv(DATA / "processed" / "standardization_log.csv")
detect = pd.read_csv(DATA / "processed" / "detection_log_v3_pilot.csv")
split_ids = pd.read_csv(REPO / "split_ids.csv")

df = clean.merge(
    split_ids[["name", "split", "person_id", "is_before", "pair_complete"]],
    on="name", how="left",
)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


# 1. Overall BMI distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
bmi = df["bmi"].astype(float)
ax.hist(bmi, bins=40, color="steelblue", edgecolor="white")
ax.axvline(bmi.mean(),   color="black", linestyle="--", label=f"mean = {bmi.mean():.2f}")
ax.axvline(bmi.median(), color="red",   linestyle="--", label=f"median = {bmi.median():.2f}")
for cutoff, label in [(18.5, "normal"), (25, "overweight"), (30, "obese")]:
    ax.axvline(cutoff, color="gray", linestyle=":", alpha=0.5)
    ax.text(cutoff, ax.get_ylim()[1] * 0.95, f" {label}", color="gray",
            fontsize=8, va="top", rotation=90)
ax.set_xlabel("BMI")
ax.set_ylabel("count")
ax.set_title(f"Overall BMI distribution (n={bmi.notna().sum()}) — right-skewed")
ax.legend()
save(fig, "bmi_distribution.png")

# 2. Missing-image bias check (BMI present vs missing)
present = audit.loc[audit["exists"]]
missing = audit.loc[~audit["exists"]]

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(15, 60, 46)
ax.hist(present["bmi"], bins=bins, alpha=0.55, density=True,
        color="steelblue", label=f"present (n={len(present)})")
ax.hist(missing["bmi"], bins=bins, alpha=0.75, density=True,
        color="crimson",   label=f"missing (n={len(missing)})")
ax.set_xlabel("BMI")
ax.set_ylabel("density")
ax.set_title("BMI distribution — present vs missing image files")
ax.legend()
save(fig, "missing_bmi_comparison.png")

# 3. BMI by split (boxplot)
fig, ax = plt.subplots(figsize=(7, 4.5))
order = ["train", "test"]
data = [df.loc[df["split"] == s, "bmi"].dropna() for s in order]
bp = ax.boxplot(data, tick_labels=[f"{s}\n(n={len(d)})" for s, d in zip(order, data)],
                patch_artist=True, showfliers=True)
for patch, color in zip(bp["boxes"], ["lightsteelblue", "lightsalmon"]):
    patch.set_facecolor(color)
ax.set_ylabel("BMI")
ax.set_title("BMI by canonical split — distributions overlap")
save(fig, "bmi_by_split.png")

# 4. Gender by split (grouped bar with percentages)
ct = pd.crosstab(df["split"], df["gender"]).reindex(["train", "test"])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(ct.index))
w = 0.38
b1 = ax.bar(x - w / 2, ct["Female"], w, color="salmon",    label="Female")
b2 = ax.bar(x + w / 2, ct["Male"],   w, color="steelblue", label="Male")
for bars, col in [(b1, "Female"), (b2, "Male")]:
    for bar, sp in zip(bars, ct.index):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(bar.get_height())}\n({ct_pct.loc[sp, col]:.1f}%)",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(ct.index)
ax.set_ylabel("count")
ax.set_title("Gender by canonical split — ratio preserved without stratification")
ax.legend()
ax.set_ylim(0, ct.values.max() * 1.18)
save(fig, "gender_by_split.png")

# 5. Aspect ratio of raw images (motivates pad-resize)
img = audit.loc[audit["exists"]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(img["aspect_ratio"], bins=40, color="seagreen", edgecolor="white")
axes[0].axvline(1.0, color="red", linestyle="--", label="square (w/h = 1)")
axes[0].set_xlabel("aspect ratio (width / height)")
axes[0].set_ylabel("count")
axes[0].set_title("Raw image aspect ratios — far from square")
axes[0].legend()

axes[1].scatter(img["width"], img["height"], s=4, alpha=0.25, color="steelblue")
lim = max(img["width"].max(), img["height"].max())
axes[1].plot([0, lim], [0, lim], "r--", linewidth=1, label="w = h")
axes[1].set_xlabel("width (px)")
axes[1].set_ylabel("height (px)")
axes[1].set_title("Width vs height — varied dimensions")
axes[1].legend()

save(fig, "aspect_ratio.png")

# 6. Pipeline funnel
n_raw_rows = len(raw)
n_clean = len(clean)
n_std_ok = (standardize["status"] == "ok").sum()
n_det_ok = (detect["status"] == "ok").sum()

stages = ["Raw rows", "Files exist", "Standardized OK", "MTCNN-aligned OK"]
vals = [n_raw_rows, n_clean, n_std_ok, n_det_ok]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(stages[::-1], vals[::-1], color="steelblue")
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 30, b.get_y() + b.get_height() / 2,
            f"{v:,}  ({v / n_raw_rows:.1%})", va="center")
ax.set_xlim(0, max(vals) * 1.18)
ax.set_title("Pipeline funnel — retention vs raw label rows")
ax.set_xlabel("images")
save(fig, "pipeline_funnel.png")

# 7. Sample pipeline strip (raw -> standardized -> MTCNN aligned)
RAW_IMG = DATA / "raw" / "Images"
STD_IMG = DATA / "processed" / "faces_standardized"
ALIGN_V3 = DATA / "processed" / "faces_aligned_v3_pilot"

available = df.loc[df["name"].apply(lambda n: (STD_IMG / n).exists())]
sample = available.sample(n=4, random_state=7).reset_index(drop=True)
stages_strip = [("Raw", RAW_IMG), ("Standardized 224x224", STD_IMG), ("MTCNN aligned (v3)", ALIGN_V3)]

fig, axes = plt.subplots(len(sample), len(stages_strip),
                         figsize=(len(stages_strip) * 2.6, len(sample) * 2.6))
for r, row in sample.iterrows():
    for c, (label, folder) in enumerate(stages_strip):
        ax = axes[r, c]
        path = folder / row["name"]
        if path.exists():
            ax.imshow(np.asarray(Image.open(path).convert("RGB")))
        else:
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if r == 0:
            ax.set_title(label, fontsize=10)
    axes[r, 0].set_ylabel(f"BMI {row['bmi']:.1f}\n{row['gender']}",
                          rotation=0, labelpad=42, fontsize=9, va="center")
fig.suptitle("Preprocessing pipeline: raw → standardized → MTCNN-aligned", y=1.00)
save(fig, "sample_pipeline.png")

print("done.")
