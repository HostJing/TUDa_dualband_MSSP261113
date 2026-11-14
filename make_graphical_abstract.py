# -*- coding: utf-8 -*-
"""Graphical abstract for the MSSP submission (Elsevier ~2.5:1 ratio)."""
import os
import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {"blue": "#3B4992", "red": "#EE0000", "green": "#008B45",
     "purple": "#631879", "teal": "#008280", "dark": "#1A1A1A"}

fig, ax = plt.subplots(figsize=(9.6, 3.6), dpi=200)
ax.set_xlim(0, 96)
ax.set_ylim(0, 36)
ax.axis("off")


def box(x, y, w, h, title, body, color, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.6,rounding_size=1.2",
                 ec=color, fc=fc, lw=1.6))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top",
            fontsize=8.6, color=color, fontweight="bold", wrap=True)
    ax.text(x + w / 2, y + h - 8.5, body, ha="center", va="top",
            fontsize=7.0, color=C["dark"], linespacing=1.45)


def arrow(x0, x1, y=18):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=16, color=C["dark"], lw=1.6))


# five-stage pipeline
box(1.0, 10, 16.5, 16, "1 | Measurement",
    "500 kHz casing array,\n26 steady points,\n2 stall runs\n(TUDa transonic)", C["blue"])
box(20.3, 10, 17.5, 16, "2 | Normalization",
    "stall band\n0.25–0.625 $f_R$,\nsurge band\n0.001–0.03 $f_R$", C["teal"])
box(40.6, 10, 17.5, 16, "3 | Calibration",
    "POT fit on steady\npoints → $K=2.195$,\nzero false alarms,\nno manual threshold", C["green"])
box(60.9, 10, 17.5, 16, "4 | Detection",
    "zero retuning:\nN100 & N65 alarms\nwithin 0.13 s\nof the onset", C["purple"])
box(81.2, 10, 14.0, 16, "5 | Value",
    "precursor lead\n0.72–0.92 s >\n0.55 s required\nfor avoidance", C["red"])

for x0, x1 in ((18.3, 20.0), (38.6, 40.3), (58.9, 60.6), (79.2, 80.9)):
    arrow(x0, x1)

# top banner
ax.text(48, 32.5, "Rotor-frequency-normalized dual-band detection of rotating-stall inception",
        ha="center", fontsize=11.5, color=C["dark"], fontweight="bold")
ax.text(48, 28.8, "zero-retuning cross-speed validation on the open TUDa-GLR-OpenStage dataset",
        ha="center", fontsize=8.6, color="#555555", style="italic")

# bottom strip: DL contrast
ax.add_patch(FancyBboxPatch((14, 1.2), 68, 6.2,
             boxstyle="round,pad=0.5,rounding_size=1.0",
             ec="#999999", fc="#F2F2F2", lw=1.0, linestyle="--"))
ax.text(48, 5.6, "Learned baselines under the same task: CNN/LSTM reach AP 0.78 in distribution,",
        ha="center", fontsize=7.4, color="#555555")
ax.text(48, 2.9, "but stay in permanent alarm or lose all early warning after a change of speed.",
        ha="center", fontsize=7.4, color="#555555")

fig.tight_layout(pad=0.4)
OUT_DIR = os.path.join(ROOT, "paper", "submission")
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(os.path.join(OUT_DIR, "graphical_abstract.pdf"),
            bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "graphical_abstract.png"),
            bbox_inches="tight")
print("graphical abstract saved ->", OUT_DIR)
