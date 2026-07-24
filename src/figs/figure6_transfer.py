"""
figs/figure6_transfer.py
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


class Figure6Transfer:
    DIAG_CSV = "inputs/results/diagnostics.csv"
    ROBUST_CSV = "inputs/results/robustness.csv"
    OUT_PDF = "figs/figure6_transfer.pdf"

    MOD_COLORS = {"images": "#2166AC", "radtext": "#D6604D", "histnote": "#E08214"}
    MOD_LABELS = {"images": "Chest radiograph", "radtext": "Radiology report",
                  "histnote": "Prior discharge summary"}
    EMU_ORDER = ["fluids_sepsis", "transfusion_threshold", "rrt_timing", "prone_positioning"]
    EMU_LABELS = {"fluids_sepsis": "Fluid strategy", "transfusion_threshold": "Transfusion",
                  "rrt_timing": "RRT timing", "prone_positioning": "Proning"}
    EMU_MARKERS = {"fluids_sepsis": "o", "transfusion_threshold": "s",
                   "rrt_timing": "^", "prone_positioning": "D"}
    DS_ORDER = ["images@padchest", "images@chestxray14", "images@chexpert"]
    DS_LABELS = {"images@padchest": "PadChest", "images@chestxray14": "ChestX-ray14",
                 "images@chexpert": "CheXpert"}
    FIND_ORDER = ["edema", "effusion", "cardiomegaly"]
    SWAP_ORDER = ["primary", "encoder", "window", "pooling", "estimator", "reduction", "trim"]
    SWAP_LABELS = {"primary": "Primary", "encoder": "Encoder", "window": "Window",
                   "pooling": "Pooling", "estimator": "Estimator",
                   "reduction": "Reduction", "trim": "Trimming"}
    EXTERNAL = "#1B7837"
    DEPLOY = "#2166AC"
    REF = "#333333"
    AUC_THRESHOLD = 2.0

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
            "axes.labelsize": 16.5,
            "axes.titlesize": 18.5,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
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
        self.ext = d[d["check"] == "informativeness_external"].copy()
        self.full = d[d["check"] == "full"].copy()
        rb = pd.read_csv(self.ROBUST_CSV)
        self.sw = rb[rb["family"] == "swap"].copy()

    @staticmethod
    def _clean(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    @staticmethod
    def _panel_label(ax, letter, title, x=-0.16, y=1.045, offset=0.080):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=24,
                fontweight="bold", ha="left", va="bottom")
        ax.text(x + offset, y, title, transform=ax.transAxes, fontsize=17.5,
                ha="left", va="bottom")

    def _panel_a(self, ax):
        self._clean(ax)
        w = 0.26
        x = np.arange(len(self.DS_ORDER))
        for j, f in enumerate(self.FIND_ORDER):
            vals, errs = [], []
            for ds in self.DS_ORDER:
                row = self.ext[(self.ext["modality"] == ds) &
                               (self.ext["probe_target"] == f)].iloc[0]
                vals.append(row["probe_auroc_mean"])
                errs.append(row["probe_auroc_std"])
            ax.bar(x + (j - 1) * w, vals, width=w, yerr=errs, capsize=3,
                   color=self.EXTERNAL, alpha=0.35 + 0.28 * j, edgecolor="white",
                   linewidth=0.9, zorder=2, error_kw=dict(elinewidth=1.2, ecolor=self.REF),
                   label=f.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels([self.DS_LABELS[d] for d in self.DS_ORDER], fontsize=13.0)
        ax.set_ylim(50, 100)
        ax.set_ylabel("AUROC (%)")
        ax.legend(frameon=False, loc="upper right", fontsize=13.5, ncol=1,
                  handletextpad=0.4, columnspacing=0.9)
        self._panel_label(ax, "a", "The encoder reads the findings on benchmarks")

    def _panel_b(self, ax):
        self._clean(ax)
        for _, row in self.ext.iterrows():
            ax.scatter(row["probe_auroc_mean"], row["probe_auprc_mean"], s=150,
                       color=self.EXTERNAL, alpha=0.85, edgecolor="white", linewidth=1.0,
                       zorder=3)
        worst = self.ext.loc[self.ext["probe_auprc_mean"].idxmin()]
        ax.annotate(f"{self.DS_LABELS[worst['modality']]}, {worst['probe_target']}\n"
                    f"AUROC {worst['probe_auroc_mean']:.1f}, AUPRC {worst['probe_auprc_mean']:.1f}",
                    (worst["probe_auroc_mean"], worst["probe_auprc_mean"]),
                    textcoords="offset points", xytext=(-46, 92), ha="left", fontsize=13.0,
                    color=self.REF, arrowprops=dict(arrowstyle="-", color=self.REF, lw=1.0))
        ax.set_xlabel("AUROC (%)")
        ax.set_ylabel("AUPRC (%)")
        ax.set_ylim(0, 75)
        self._panel_label(ax, "b", "Discrimination overstates rare findings")

    def _panel_c(self, ax):
        self._clean(ax)
        e = self.ext[self.ext["probe_target"] == "edema"]
        f = self.full[self.full["modality"] == "images"]
        rng = np.random.default_rng(0)
        for k, (vals, col, lab) in enumerate([(e["probe_auroc_mean"].values, self.EXTERNAL,
                                               "external"),
                                              (f["probe_auroc_mean"].values, self.DEPLOY,
                                               "deployment")]):
            ax.scatter(k + rng.normal(0, 0.045, len(vals)), vals, s=150, color=col,
                       edgecolor="white", linewidth=1.0, zorder=3)
            ax.hlines(np.median(vals), k - 0.28, k + 0.28, color=col, lw=3.4, zorder=4)
            ax.text(k, np.median(vals) + 1.6, f"median {np.median(vals):.1f}", ha="center",
                    va="bottom", fontsize=13.5, color=col)
        m1, m2 = np.median(e["probe_auroc_mean"]), np.median(f["probe_auroc_mean"])
        ax.annotate("", xy=(0.62, m2), xytext=(0.62, m1),
                    arrowprops=dict(arrowstyle="<->", color=self.REF, lw=1.4))
        ax.text(0.68, (m1 + m2) / 2, f"{m1 - m2:.1f}\npoints", fontsize=13.5,
                color=self.REF, ha="left", va="center")
        ax.set_xlim(-0.5, 1.75)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Benchmark\n$n$ = {len(e)} datasets",
                            f"Deployment\n$n$ = {len(f)} cohorts"])
        ax.set_ylabel("Edema AUROC (%)")
        self._panel_label(ax, "c", "The same readout falls in deployment")

    def _panel_d(self, ax):
        self._clean(ax)
        w = 0.26
        x = np.arange(len(self.EMU_ORDER))
        for j, m in enumerate(self.MOD_COLORS):
            vals, errs = [], []
            for emu in self.EMU_ORDER:
                row = self.full[(self.full["intervention"] == emu) &
                                (self.full["modality"] == m)].iloc[0]
                vals.append(row["probe_auroc_mean"])
                errs.append(row["probe_auroc_std"])
            ax.bar(x + (j - 1) * w, vals, width=w, yerr=errs, capsize=3,
                   color=self.MOD_COLORS[m], alpha=0.88, edgecolor="white", linewidth=0.9,
                   zorder=2, error_kw=dict(elinewidth=1.2, ecolor=self.REF))
        ax.axhline(50, color=self.REF, lw=1.2, ls=":", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([self.EMU_LABELS[e].replace(" ", "\n") for e in self.EMU_ORDER],
                           fontsize=12.5)
        ax.set_ylim(48, 80)
        ax.set_ylabel("In-cohort AUROC (%)")
        self._panel_label(ax, "d", "Every channel is informative in cohort")

    def _panel_e(self, ax):
        self._clean(ax)
        for _, row in self.full.iterrows():
            ax.scatter(row["probe_auroc_mean"], row["ici"], s=150,
                       marker=self.EMU_MARKERS[row["intervention"]],
                       color=self.MOD_COLORS[row["modality"]], edgecolor="white",
                       linewidth=1.0, zorder=3)
        ax.axhline(self.AUC_THRESHOLD, color=self.REF, lw=1.5, ls="--", zorder=2)
        ax.axhline(0, color=self.REF, lw=1.0, ls=":", zorder=1)
        ax.text(self.full["probe_auroc_mean"].max(), self.AUC_THRESHOLD + 0.08,
                "threshold 2.0", fontsize=13.5, color=self.REF, ha="right", va="bottom")
        ax.set_ylim(-0.8, 2.6)
        ax.set_xlabel("In-cohort readout AUROC (%)")
        ax.set_ylabel("Index (AUROC points)")
        self._panel_label(ax, "e", "Readout does not become increment")

    def _panel_f(self, ax):
        self._clean(ax)
        est = [e for e in self.EMU_ORDER if e != "prone_positioning"]
        y = np.arange(len(est))
        for k, emu in enumerate(est):
            for j, (sw, col, off) in enumerate([("primary", self.REF, 0.16),
                                                ("encoder", self.DEPLOY, -0.16)]):
                row = self.sw[(self.sw["swap"] == sw) &
                              (self.sw["intervention"] == emu)].iloc[0]
                ax.errorbar(row["value_point"], k + off,
                            xerr=[[row["value_point"] - row["value_ci_low"]],
                                  [row["value_ci_high"] - row["value_point"]]],
                            fmt="none", ecolor=col, elinewidth=2.0, capsize=4, zorder=3)
                ax.scatter(row["value_point"], k + off, s=120, color=col, edgecolor="white",
                           linewidth=0.9, zorder=4,
                           label=("Primary encoder" if sw == "primary" else
                                  "Alternative encoder") if k == 0 else None)
        ax.axvline(0, color=self.REF, lw=1.5, ls="--", zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([self.EMU_LABELS[e] for e in est])
        ax.set_ylim(-0.6, len(est) + 0.35)
        ax.legend(frameon=False, loc="upper center", fontsize=13.5, ncol=2,
                  handletextpad=0.4, columnspacing=1.2)
        ax.set_xlabel("Paired change in distance (pp)")
        self._panel_label(ax, "f", "A second encoder gives the same null",
                          x=-0.303, offset=0.076)

    def _swap_forest(self, ax, emus, letter, title, xlab=True,
                     label_x=-0.261, label_offset=0.048):
        self._clean(ax)
        rows = []
        for sw in self.SWAP_ORDER:
            for emu in emus:
                r = self.sw[(self.sw["swap"] == sw) & (self.sw["intervention"] == emu)]
                if not r.empty:
                    rows.append((sw, emu, r.iloc[0]))
        y = np.arange(len(rows))
        for k, (sw, emu, row) in enumerate(rows):
            col = self.DEPLOY if sw == "encoder" else self.REF
            ax.errorbar(row["value_point"], k,
                        xerr=[[row["value_point"] - row["value_ci_low"]],
                              [row["value_ci_high"] - row["value_point"]]],
                        fmt="none", ecolor=col, elinewidth=1.5, capsize=2.5, zorder=3)
            ax.scatter(row["value_point"], k, s=70, marker=self.EMU_MARKERS[emu],
                       color=col, edgecolor="white", linewidth=0.7, zorder=4)
        ax.axvline(0, color=self.REF, lw=1.5, ls="--", zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{self.SWAP_LABELS[sw]}, {self.EMU_LABELS[emu].split()[0]}"
                            for sw, emu, _ in rows], fontsize=11.8)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        n = len(rows)
        ax.text(0.97, 0.02, f"all {n} of {n} span zero", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=13.0, color=self.REF, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.0))
        if xlab:
            ax.set_xlabel("Paired change in distance (pp)")
        self._panel_label(ax, letter, title, x=label_x, offset=label_offset)

    def _panel_g(self, ax):
        self._swap_forest(ax, [e for e in self.EMU_ORDER if e != "prone_positioning"],
                          "g", "No design choice changes the answer",
                          label_x=-0.261, label_offset=0.048)

    def _panel_h(self, ax):
        self._swap_forest(ax, ["prone_positioning"], "h",
                          "Proning is uninformative under every swap",
                          label_x=-0.340, label_offset=0.075)

    def _legend(self, ax):
        ax.axis("off")
        h = [Line2D([0], [0], marker="o", lw=0, ms=13, mfc=self.EXTERNAL, mec="white",
                    label="Public benchmark datasets"),
             Line2D([0], [0], marker="o", lw=0, ms=13, mfc=self.DEPLOY, mec="white",
                    label="Deployment cohorts and alternative encoder")]
        h += [Line2D([0], [0], marker="o", lw=0, ms=13, mfc=self.MOD_COLORS[m], mec="white",
                     label=self.MOD_LABELS[m]) for m in self.MOD_COLORS]
        h += [Line2D([0], [0], ls="--", lw=1.6, color=self.REF, label="Decision threshold")]
        ax.legend(handles=h, ncol=3, frameon=False, loc="center", columnspacing=1.8,
                  handletextpad=0.55, fontsize=15.0)

    def build(self):
        self.fig = plt.figure(figsize=(19.0, 20.5), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.995, bottom=0.930)
        gs_r1 = self.fig.add_gridspec(1, 3, left=0.070, right=0.980, top=0.885, bottom=0.650,
                                      wspace=0.36)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.070, right=0.980, top=0.578, bottom=0.352,
                                      width_ratios=[1.0, 1.0, 1.10], wspace=0.44)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.133, right=0.975, top=0.278, bottom=0.040,
                                      width_ratios=[1.55, 1.0], wspace=0.60)
        self._legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_a(self.fig.add_subplot(gs_r1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs_r1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs_r1[0, 2]))
        self._panel_d(self.fig.add_subplot(gs_r2[0, 0]))
        self._panel_e(self.fig.add_subplot(gs_r2[0, 1]))
        self._panel_f(self.fig.add_subplot(gs_r2[0, 2]))
        self._panel_g(self.fig.add_subplot(gs_r3[0, 0]))
        self._panel_h(self.fig.add_subplot(gs_r3[0, 1]))
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
    Figure6Transfer().build().save()
