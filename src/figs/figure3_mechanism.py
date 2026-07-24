"""
figs/figure3_mechanism.py
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
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


class Figure3Mechanism:
    SYNTH_CSV = "inputs/results/synthetic.csv"
    OUT_PDF = "figs/figure3_mechanism.pdf"

    PRIMARY = "#2166AC"
    SECOND = "#4393C3"
    NULLGRAY = "#777777"
    REAL = "#E08214"
    VETO = "#B2182B"
    REF = "#333333"

    AUC_THRESHOLD = 2.0
    COST_CAP = 0.25
    PP_BAND = 2.0

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
        s = pd.read_csv(self.SYNTH_CSV)
        self.red = s[s["sweep"] == "redundancy"].copy()
        self.grid = s[s["sweep"] == "gamma_delta"].copy()
        self.ctrl = self.grid[(self.grid["gamma"] == 1.5) & (self.grid["delta"] == 1.5)]
        self.rhos = np.sort(self.red["redundancy"].unique())
        self.lvl = (self.red.groupby("redundancy")
                    .agg(bs=("bias_structured", "mean"), bs_sd=("bias_structured", "std"),
                         bi=("bias_struct_img", "mean"), bi_sd=("bias_struct_img", "std"),
                         br=("bias_reduction", "mean"), br_sd=("bias_reduction", "std"),
                         dt=("d_auc_treat", "mean"), dt_sd=("d_auc_treat", "std"),
                         do=("d_auc_outcome", "mean"), do_sd=("d_auc_outcome", "std"),
                         ici=("ici", "mean"), ici_sd=("ici", "std"),
                         pc=("positivity_cost", "mean"), pc_sd=("positivity_cost", "std"))
                    .reset_index())
        self.cmap = LinearSegmentedColormap.from_list("rho", [self.PRIMARY, self.NULLGRAY])

    def _rho_color(self, rho):
        span = self.rhos.max() - self.rhos.min()
        return self.cmap((rho - self.rhos.min()) / span if span else 0.0)

    @staticmethod
    def _clean(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    @staticmethod
    def _panel_label(ax, letter, title, x=-0.17, y=1.045, offset=0.095):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=24,
                fontweight="bold", ha="left", va="bottom")
        ax.text(x + offset, y, title, transform=ax.transAxes, fontsize=17.5,
                ha="left", va="bottom")

    def _band(self, ax, x, m, sd, color, label=None, ls="-", marker="o"):
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.16, zorder=1)
        ax.plot(x, m, ls, color=color, lw=2.6, marker=marker, ms=10,
                mec="white", mew=1.0, zorder=3, label=label)

    def _panel_a(self, ax):
        self._clean(ax)
        rng = np.random.default_rng(0)
        ax.axhspan(-self.PP_BAND, self.PP_BAND, color=self.REF, alpha=0.10, zorder=0)
        ax.axhline(0, color=self.REF, lw=1.0, ls=":", zorder=1)
        for rho in self.rhos:
            v = self.red[self.red["redundancy"] == rho]["bias_reduction"].values
            ax.scatter(rho + rng.normal(0, 0.016, len(v)), v, s=26,
                       color=self._rho_color(rho), alpha=0.55, linewidths=0, zorder=2)
        self._band(ax, self.lvl["redundancy"], self.lvl["br"], self.lvl["br_sd"], self.PRIMARY)
        rho_s, p_s = stats.spearmanr(self.red["redundancy"], self.red["bias_reduction"])
        ax.text(0.97, 0.95, f"$\\rho_s$ = {rho_s:.3f}\n$p$ = {p_s:.1e}", transform=ax.transAxes,
                ha="right", va="top", fontsize=14.0, color=self.REF)
        ax.set_xlabel(r"Redundancy $\rho$")
        ax.set_ylabel("Bias reduction (pp)")
        self._panel_label(ax, "a", "Bias reduction collapses as redundancy rises")

    def _panel_b(self, ax):
        self._clean(ax)
        self._band(ax, self.lvl["redundancy"], self.lvl["bs"], self.lvl["bs_sd"],
                   self.REF, label="structured", marker="s")
        self._band(ax, self.lvl["redundancy"], self.lvl["bi"], self.lvl["bi_sd"],
                   self.PRIMARY, label="structured $+$ image")
        ax.axhline(0, color=self.REF, lw=1.0, ls=":", zorder=1)
        ax.legend(frameon=False, loc="upper right", fontsize=14.5, handletextpad=0.5)
        ax.set_xlabel(r"Redundancy $\rho$")
        ax.set_ylabel("Bias (pp)")
        self._panel_label(ax, "b", "Both biases fall and the gap closes")

    def _panel_c(self, ax):
        self._clean(ax)
        share = 100.0 * self.lvl["br"] / self.lvl["bs"]
        ax.plot(self.lvl["redundancy"], share, "-o", color=self.PRIMARY, lw=2.6, ms=11,
                mec="white", mew=1.0, zorder=3)
        for x, y in zip(self.lvl["redundancy"], share):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=13.0, color=self.PRIMARY)
        ax.set_ylim(0, share.max() * 1.28)
        ax.set_xlabel(r"Redundancy $\rho$")
        ax.set_ylabel("Structured bias removed (%)")
        self._panel_label(ax, "c", "The share removed falls in step")

    def _panel_d(self, ax):
        self._clean(ax)
        self._band(ax, self.lvl["redundancy"], self.lvl["ici"], self.lvl["ici_sd"], self.PRIMARY)
        ax.axhline(self.AUC_THRESHOLD, color=self.REF, lw=1.5, ls="--", zorder=2)
        ax.text(self.rhos.max(), self.AUC_THRESHOLD + 0.22, "threshold 2.0", fontsize=13.5,
                color=self.REF, ha="right", va="bottom")
        ax.axhline(0, color=self.REF, lw=1.0, ls=":", zorder=1)
        rho_s, p_s = stats.spearmanr(self.red["redundancy"], self.red["ici"])
        ax.text(0.97, 0.95, f"$\\rho_s$ = {rho_s:.3f}\n$p$ = {p_s:.1e}", transform=ax.transAxes,
                ha="right", va="top", fontsize=14.0, color=self.REF)
        ax.set_xlabel(r"Redundancy $\rho$")
        ax.set_ylabel("Index (AUROC points)")
        self._panel_label(ax, "d", "The index tracks the same collapse")

    def _panel_e(self, ax):
        self._clean(ax)
        self._band(ax, self.lvl["redundancy"], self.lvl["dt"], self.lvl["dt_sd"],
                   self.PRIMARY, label="treatment")
        self._band(ax, self.lvl["redundancy"], self.lvl["do"], self.lvl["do_sd"],
                   self.SECOND, label="outcome", marker="^")
        ax.axhline(self.AUC_THRESHOLD, color=self.REF, lw=1.5, ls="--", zorder=2)
        ax.axhline(0, color=self.REF, lw=1.0, ls=":", zorder=1)
        ax.legend(frameon=False, loc="upper right", fontsize=14.5, handletextpad=0.5,
                  title="increment", title_fontsize=13.5)
        ax.set_xlabel(r"Redundancy $\rho$")
        ax.set_ylabel("Increment (AUROC points)")
        self._panel_label(ax, "e", "Both increments fall together")

    def _panel_f(self, ax):
        self._clean(ax)
        for rho in self.rhos:
            sub = self.red[self.red["redundancy"] == rho]
            ax.scatter(sub["ici"], sub["bias_reduction"], s=46, color=self._rho_color(rho),
                       alpha=0.80, edgecolor="white", linewidth=0.5, zorder=3)
        ax.axvline(self.AUC_THRESHOLD, color=self.REF, lw=1.4, ls="--", zorder=2)
        ax.axhline(0, color=self.REF, lw=1.0, ls=":", zorder=1)
        rho_s, p_s = stats.spearmanr(self.red["ici"], self.red["bias_reduction"])
        ax.text(0.97, 0.06, f"$\\rho_s$ = {rho_s:.3f}\n$p$ = {p_s:.1e}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=14.0, color=self.REF)
        sm = plt.cm.ScalarMappable(cmap=self.cmap,
                                   norm=plt.Normalize(self.rhos.min(), self.rhos.max()))
        cb = ax.figure.colorbar(sm, ax=ax, fraction=0.045, pad=0.03, ticks=self.rhos)
        cb.ax.tick_params(labelsize=12.5)
        cb.set_label(r"redundancy $\rho$", fontsize=13.5)
        ax.set_xlabel("Index (AUROC points)")
        ax.set_ylabel("Bias reduction (pp)")
        self._panel_label(ax, "f", "The index predicts the collapse run by run")

    def _panel_g(self, ax):
        self._clean(ax)
        self._band(ax, self.lvl["redundancy"], self.lvl["pc"], self.lvl["pc_sd"], self.PRIMARY)
        ax.axhline(self.COST_CAP, color=self.VETO, lw=1.6, ls="--", zorder=2)
        ax.text(self.rhos.max(), self.COST_CAP + 0.012, "cap 0.25", fontsize=13.5,
                color=self.VETO, ha="right", va="bottom")
        ax.axhline(0, color=self.REF, lw=1.0, ls=":", zorder=1)
        for i in (0, len(self.lvl) - 1):
            ax.annotate(f"{self.lvl['pc'].iloc[i]:.3f}",
                        (self.lvl["redundancy"].iloc[i], self.lvl["pc"].iloc[i]),
                        textcoords="offset points", xytext=(10, 10), fontsize=13.0,
                        color=self.PRIMARY)
        ax.set_xlabel(r"Redundancy $\rho$")
        ax.set_ylabel("Positivity cost")
        self._panel_label(ax, "g", "A fully redundant modality is also free", x=-0.13, offset=0.070)

    def _panel_h(self, ax):
        self._clean(ax)
        z = self.red[self.red["redundancy"] == 0.0]
        c = self.ctrl
        rng = np.random.default_rng(1)
        sets = [(r"redundancy sweep, $\rho = 0$", z, self.PRIMARY, "o"),
                (r"calibration grid, $\gamma = \delta = 1.5$", c, self.SECOND, "s")]
        for k, (name, sub, col, mk) in enumerate(sets):
            v = sub["bias_reduction"].values
            parts = ax.violinplot([v], positions=[k], widths=0.68, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(col)
                b.set_alpha(0.28)
                b.set_edgecolor(col)
            ax.scatter(k + rng.normal(0, 0.05, len(v)), v, s=30, color=col, marker=mk,
                       alpha=0.65, linewidths=0, zorder=2)
            ax.hlines(v.mean(), k - 0.34, k + 0.34, color=col, lw=3.2, zorder=4)
            side = -1 if k == 0 else 1
            ax.text(k + side * 0.40, v.mean(), f"{v.mean():.1f} (SD {v.std(ddof=1):.1f})\n"
                                               f"index {sub['ici'].mean():.2f}",
                    ha="right" if side < 0 else "left", va="center", fontsize=13.5,
                    color=col, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.6))
        w = stats.ttest_ind(z["bias_reduction"], c["bias_reduction"], equal_var=False)
        ax.text(0.5, 0.03, f"Welch $t$ = {w.statistic:.3f}, $p$ = {w.pvalue:.3f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=14.0, color=self.REF)
        ax.set_xlim(-1.05, 1.90)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"$\\rho = 0$\n$n$ = {len(z)}", f"grid cell\n$n$ = {len(c)}"])
        ax.set_ylabel("Bias reduction (pp)")
        self._panel_label(ax, "h", "The two sweeps agree where they meet", x=-0.13, offset=0.070)

    def _legend(self, ax):
        ax.axis("off")
        handles = [
            Line2D([0], [0], marker="o", lw=0, ms=14, mfc=self.PRIMARY, mec="white",
                   label=r"Confounder outside the record ($\rho = 0$)"),
            Line2D([0], [0], marker="o", lw=0, ms=14, mfc=self.NULLGRAY, mec="white",
                   label=r"Confounder inside the record ($\rho = 1$)"),
            Line2D([0], [0], marker="s", lw=2.6, ms=11, color=self.REF, mfc=self.REF,
                   mec="white", label="Structured adjustment"),
            Line2D([0], [0], marker="^", lw=2.6, ms=11, color=self.SECOND, mfc=self.SECOND,
                   mec="white", label="Outcome-side increment"),
            Line2D([0], [0], ls="--", lw=1.6, color=self.REF, label="Decision threshold"),
        ]
        ax.legend(handles=handles, ncol=5, frameon=False, loc="center",
                  columnspacing=1.7, handletextpad=0.55, fontsize=15.0)

    def build(self):
        self.fig = plt.figure(figsize=(19.0, 19.0), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.995, bottom=0.958)
        gs_r1 = self.fig.add_gridspec(1, 3, left=0.070, right=0.980, top=0.900, bottom=0.652,
                                      wspace=0.33)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.070, right=0.980, top=0.578, bottom=0.330,
                                      wspace=0.33)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.256, bottom=0.048,
                                      wspace=0.24)
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
    Figure3Mechanism().build().save()
