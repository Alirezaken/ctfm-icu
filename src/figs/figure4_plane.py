"""
figs/figure4_plane.py
Created on July 20, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Figure4Plane:
    DIAG_CSV = "inputs/results/diagnostics.csv"
    OUT_PDF = "figs/figure4_plane.pdf"

    MOD_COLORS = {"images": "#2166AC", "radtext": "#D6604D", "histnote": "#E08214"}
    MOD_LABELS = {"images": "Chest radiograph", "radtext": "Radiology report",
                  "histnote": "Prior discharge summary"}
    EMU_MARKERS = {"fluids_sepsis": "o", "transfusion_threshold": "s",
                   "rrt_timing": "^", "prone_positioning": "D"}
    EMU_LABELS = {"fluids_sepsis": "Fluid strategy", "transfusion_threshold": "Transfusion",
                  "rrt_timing": "RRT timing", "prone_positioning": "Proning"}
    QUAD_COLORS = {"redundant": "#BBBBBB", "precision variable only": "#E08214",
                   "instrument-like": "#B2182B", "accept": "#1B7837"}
    REF = "#333333"
    VETO = "#B2182B"

    AUC_THRESHOLD = 2.0
    COST_CAP = 0.25

    def __init__(self, out_pdf=None):
        self.out_pdf = Path(out_pdf or self.OUT_PDF)
        self._set_rcparams()
        self._load_data()
        self.fig = None

    @staticmethod
    def _set_rcparams():
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 17.0,
            "axes.labelsize": 17.0,
            "axes.titlesize": 18.5,
            "xtick.labelsize": 14.5,
            "ytick.labelsize": 14.5,
            "legend.fontsize": 15.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.15,
            "xtick.major.width": 1.05,
            "ytick.major.width": 1.05,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    def _load_data(self):
        d = pd.read_csv(self.DIAG_CSV)
        r = d[d["check"] == "full"].copy()
        r["verdict"] = r["reason"].str.split(":").str[0]
        r["gap"] = r["d_auc_outcome"] - r["d_auc_treat"]
        r["binds_treatment"] = np.isclose(r["ici"], r["d_auc_treat"])
        r["cell"] = r["intervention"].map(self.EMU_LABELS) + ", " + \
                    r["modality"].map(lambda m: self.MOD_LABELS[m].split()[0].lower())
        self.r = r
        self.emus = list(self.EMU_MARKERS)
        self.mods = list(self.MOD_COLORS)

    @staticmethod
    def _clean(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    @staticmethod
    def _panel_label(ax, letter, title, x=-0.15, y=1.045, offset=0.075):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=24,
                fontweight="bold", ha="left", va="bottom")
        ax.text(x + offset, y, title, transform=ax.transAxes, fontsize=17.5,
                ha="left", va="bottom")

    def _panel_a(self, ax):
        self._clean(ax)
        t = self.AUC_THRESHOLD
        xlo, xhi, ylo, yhi = -1.0, 4.2, -0.6, 3.6
        quads = [((xlo, ylo), t - xlo, t - ylo, "redundant", "Redundant"),
                 ((t, ylo), xhi - t, t - ylo, "instrument-like", "Instrument-like"),
                 ((xlo, t), t - xlo, yhi - t, "precision variable only", "Precision variable"),
                 ((t, t), xhi - t, yhi - t, "accept", "Accept")]
        for (x0, y0), w, h, key, lab in quads:
            ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=self.QUAD_COLORS[key],
                                       alpha=0.11, edgecolor="none", zorder=0))
            n = int((self.r["verdict"] == key).sum()) if key != "accept" else \
                int(((self.r["d_auc_treat"] >= t) & (self.r["d_auc_outcome"] >= t)).sum())
            ax.text(x0 + w - 0.10, y0 + h - 0.10, f"{lab}\n$n$ = {n}", ha="right", va="top",
                    fontsize=14.5, color=self.QUAD_COLORS[key], zorder=1)
        ax.axvline(t, color=self.REF, lw=1.5, ls="--", zorder=2)
        ax.axhline(t, color=self.REF, lw=1.5, ls="--", zorder=2)
        for _, row in self.r.iterrows():
            ax.scatter(row["d_auc_treat"], row["d_auc_outcome"], s=190,
                       marker=self.EMU_MARKERS[row["intervention"]],
                       color=self.MOD_COLORS[row["modality"]], edgecolor="white",
                       linewidth=1.2, zorder=4)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("Treatment-side increment (AUROC points)")
        ax.set_ylabel("Outcome-side increment (AUROC points)")
        self._panel_label(ax, "a", "No cell reaches the acceptable region",
                          x=-0.105, offset=0.048)

    def _panel_b(self, ax):
        self._clean(ax)
        s = self.r.sort_values("ici")
        y = np.arange(len(s))
        ax.hlines(y, 0, s["ici"], color=[self.MOD_COLORS[m] for m in s["modality"]],
                  lw=2.2, alpha=0.75, zorder=2)
        for k, (_, row) in enumerate(s.iterrows()):
            ax.scatter(row["ici"], k, s=120, marker=self.EMU_MARKERS[row["intervention"]],
                       color=self.MOD_COLORS[row["modality"]], edgecolor="white",
                       linewidth=0.9, zorder=4)
        ax.axvline(self.AUC_THRESHOLD, color=self.REF, lw=1.5, ls="--", zorder=3)
        ax.axvline(0, color=self.REF, lw=1.0, ls=":", zorder=1)
        ax.text(self.AUC_THRESHOLD - 0.06, len(s) - 0.4, "threshold 2.0", rotation=90,
                fontsize=13.5, color=self.REF, ha="right", va="top")
        ax.set_yticks(y)
        ax.set_yticklabels(s["cell"], fontsize=12.5)
        ax.set_xlim(min(-0.45, s["ici"].min() - 0.2), self.AUC_THRESHOLD + 0.45)
        ax.set_xlabel("Index (AUROC points)")
        self._panel_label(ax, "b", "Every index falls short of the threshold",
                          x=-0.361, offset=0.059)

    def _panel_c(self, ax):
        self._clean(ax)
        order = ["redundant", "precision variable only", "instrument-like", "accept"]
        labs = ["Redundant", "Precision\nvariable", "Instrument-\nlike", "Accept"]
        bottom = np.zeros(len(order))
        for m in self.mods:
            vals = [int(((self.r["verdict"] == v) & (self.r["modality"] == m)).sum())
                    if v != "accept" else 0 for v in order]
            ax.bar(range(len(order)), vals, bottom=bottom, color=self.MOD_COLORS[m],
                   edgecolor="white", linewidth=1.0, width=0.68, zorder=2)
            bottom += np.array(vals, dtype=float)
        for i, tot in enumerate(bottom):
            ax.text(i, tot + 0.14, f"{int(tot)}", ha="center", va="bottom",
                    fontsize=15.0, color=self.REF)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labs, fontsize=12.5)
        ax.set_ylim(0, max(bottom) * 1.25)
        ax.set_ylabel("Cells")
        self._panel_label(ax, "c", "Every failure mode occurs", x=-0.22, offset=0.095)

    def _panel_d(self, ax):
        self._clean(ax)
        s = self.r.sort_values("gap")
        y = np.arange(len(s))
        for k, (_, row) in enumerate(s.iterrows()):
            col = self.MOD_COLORS[row["modality"]]
            ax.plot([row["d_auc_treat"], row["d_auc_outcome"]], [k, k], color=col,
                    lw=1.8, alpha=0.65, zorder=2)
            ax.scatter(row["d_auc_treat"], k, s=95, marker="o", color="white",
                       edgecolor=col, linewidth=2.0, zorder=4)
            ax.scatter(row["d_auc_outcome"], k, s=95, marker="o", color=col,
                       edgecolor="white", linewidth=0.9, zorder=4)
        ax.axvline(self.AUC_THRESHOLD, color=self.REF, lw=1.5, ls="--", zorder=3)
        ax.axvline(0, color=self.REF, lw=1.0, ls=":", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(s["cell"], fontsize=12.5)
        ax.set_xlabel("Increment (AUROC points)")
        self._panel_label(ax, "d", "The outcome side is larger",
                          x=-0.451, offset=0.073)

    def _panel_e(self, ax):
        self._clean(ax)
        s = self.r.sort_values("gap")
        y = np.arange(len(s))
        ax.barh(y, s["gap"], color=[self.MOD_COLORS[m] for m in s["modality"]],
                height=0.62, edgecolor="white", linewidth=0.8, zorder=2)
        ax.axvline(0, color=self.REF, lw=1.3, zorder=3)
        n_pos = int((self.r["gap"] > 0).sum())
        ax.text(0.04, 0.62, f"outcome exceeds\ntreatment in {n_pos}\nof {len(self.r)} cells",
                transform=ax.transAxes, ha="left", va="center", fontsize=14.0, color=self.REF)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_xlabel("Outcome minus treatment (AUROC points)")
        ax.set_ylabel("Cells, ordered as in d")
        self._panel_label(ax, "e", "The treatment channel is what binds",
                          x=-0.16, offset=0.085)

    def _panel_f(self, ax):
        self._clean(ax)
        rng = np.random.default_rng(0)
        for i, m in enumerate(self.mods):
            sub = self.r[self.r["modality"] == m]
            for j, fld in enumerate(["d_auc_treat", "d_auc_outcome"]):
                x = i + (j - 0.5) * 0.42
                if j == 0:
                    ax.scatter(x + rng.normal(0, 0.030, len(sub)), sub[fld], s=95,
                               facecolors="none", edgecolors=self.MOD_COLORS[m],
                               linewidths=2.0, zorder=3)
                else:
                    ax.scatter(x + rng.normal(0, 0.030, len(sub)), sub[fld], s=95,
                               color=self.MOD_COLORS[m], edgecolor="white",
                               linewidth=0.8, zorder=3)
                ax.hlines(sub[fld].median(), x - 0.16, x + 0.16, color=self.MOD_COLORS[m],
                          lw=3.0, zorder=4)
        ax.axhline(self.AUC_THRESHOLD, color=self.REF, lw=1.5, ls="--", zorder=2)
        ax.axhline(0, color=self.REF, lw=1.0, ls=":", zorder=1)
        ax.set_xticks(range(len(self.mods)))
        ax.set_xticklabels([self.MOD_LABELS[m].replace(" ", "\n") for m in self.mods],
                           fontsize=13.0)
        ax.set_ylabel("Increment (AUROC points)")
        self._panel_label(ax, "f", "Radiographs are weakest on the treatment side",
                          x=-0.16, offset=0.072)

    def _panel_g(self, ax):
        self._clean(ax)
        s = self.r.sort_values("positivity_cost")
        y = np.arange(len(s))
        ax.barh(y, s["positivity_cost"], color=[self.MOD_COLORS[m] for m in s["modality"]],
                height=0.62, edgecolor="white", linewidth=0.8, zorder=2)
        ax.axvline(0, color=self.REF, lw=1.3, zorder=3)
        ax.axvline(self.COST_CAP, color=self.VETO, lw=1.6, ls="--", zorder=3)
        ax.text(self.COST_CAP, len(s) - 0.4, " cap 0.25", fontsize=13.5, color=self.VETO,
                ha="left", va="top")
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_xlim(s["positivity_cost"].min() - 0.03, self.COST_CAP + 0.09)
        ax.set_xlabel("Positivity cost")
        ax.set_ylabel("Cells")
        self._panel_label(ax, "g", "No decline is a matter of cost", x=-0.13, offset=0.062)

    def _panel_h(self, ax):
        self._clean(ax)
        base = self.r.groupby("intervention")[["auc_treat_structured",
                                               "auc_outcome_structured"]].first()
        base = base.reindex(self.emus)
        x = np.arange(len(base))
        ax.bar(x - 0.19, base["auc_treat_structured"], width=0.36, color=self.REF,
               alpha=0.72, edgecolor="white", linewidth=0.9, zorder=2, label="treatment")
        ax.bar(x + 0.19, base["auc_outcome_structured"], width=0.36, color=self.REF,
               alpha=0.36, edgecolor="white", linewidth=0.9, zorder=2, label="outcome")
        for xi, (a, b) in enumerate(zip(base["auc_treat_structured"],
                                        base["auc_outcome_structured"])):
            ax.text(xi - 0.19, a + 0.5, f"{a:.1f}", ha="center", va="bottom", fontsize=12.5)
            ax.text(xi + 0.19, b + 0.5, f"{b:.1f}", ha="center", va="bottom", fontsize=12.5)
        ax.set_xticks(x)
        ax.set_xticklabels([self.EMU_LABELS[e].replace(" ", "\n") for e in self.emus],
                           fontsize=13.0)
        ax.set_ylim(50, 82)
        ax.set_ylabel("AUROC (%)")
        ax.legend(frameon=False, loc="upper left", fontsize=14.0, ncol=2,
                  handletextpad=0.5, columnspacing=1.1)
        self._panel_label(ax, "h", "What the increments are added to",
                          x=-0.16, offset=0.072)

    def _legend(self, ax):
        ax.axis("off")
        mods = [Line2D([0], [0], marker="o", lw=0, ms=14, mfc=self.MOD_COLORS[m],
                       mec="white", label=self.MOD_LABELS[m]) for m in self.mods]
        emus = [Line2D([0], [0], marker=self.EMU_MARKERS[e], lw=0, ms=13, mfc=self.REF,
                       mec="white", label=self.EMU_LABELS[e]) for e in self.emus]
        leg1 = ax.legend(handles=mods, ncol=3, frameon=False, loc="center",
                         bbox_to_anchor=(0.24, 0.5), columnspacing=1.6,
                         handletextpad=0.5, fontsize=15.0)
        ax.add_artist(leg1)
        ax.legend(handles=emus, ncol=4, frameon=False, loc="center",
                  bbox_to_anchor=(0.74, 0.5), columnspacing=1.4,
                  handletextpad=0.5, fontsize=15.0)

    def build(self):
        self.fig = plt.figure(figsize=(19.0, 20.0), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.995, bottom=0.958)
        gs_r1 = self.fig.add_gridspec(1, 2, left=0.070, right=0.980, top=0.905, bottom=0.640,
                                      width_ratios=[1.32, 1.0], wspace=0.42)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.070, right=0.962, top=0.568, bottom=0.322,
                                      width_ratios=[0.94, 1.24, 1.02], wspace=0.50)
        gs_r3 = self.fig.add_gridspec(1, 3, left=0.070, right=0.980, top=0.250, bottom=0.045,
                                      wspace=0.36)
        self._legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_a(self.fig.add_subplot(gs_r1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs_r1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs_r2[0, 0]))
        self._panel_d(self.fig.add_subplot(gs_r2[0, 1]))
        self._panel_e(self.fig.add_subplot(gs_r2[0, 2]))
        self._panel_f(self.fig.add_subplot(gs_r3[0, 0]))
        self._panel_g(self.fig.add_subplot(gs_r3[0, 1]))
        self._panel_h(self.fig.add_subplot(gs_r3[0, 2]))
        return self

    def save(self):
        if self.fig is None:
            raise RuntimeError("Call build() before save().")
        os.makedirs(self.out_pdf.parent, exist_ok=True)
        self.fig.savefig(self.out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
        plt.close(self.fig)
        print(f"Saved: {self.out_pdf}")
        return str(self.out_pdf)


if __name__ == "__main__":
    Figure4Plane().build().save()
