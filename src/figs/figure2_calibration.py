"""
figs/figure2_calibration.py
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
from matplotlib.patches import Rectangle
from scipy import stats


class Figure2Calibration:
    SYNTH_CSV = "inputs/results/synthetic.csv"
    DIAG_CSV = "inputs/results/diagnostics.csv"
    OUT_PDF = "figs/figure2_calibration.pdf"

    PRIMARY = "#2166AC"
    NULLGRAY = "#777777"
    REAL = "#E08214"
    VETO = "#B2182B"
    REF = "#333333"

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
            "legend.fontsize": 16.0,
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
        self.grid = s[s["sweep"] == "gamma_delta"].copy()
        d = pd.read_csv(self.DIAG_CSV)
        self.real = d[d["check"] == "full"].copy()

        self.cell = (self.grid.groupby(["gamma", "delta"])
                     .agg(bias_s=("bias_structured", "mean"),
                          bias_i=("bias_struct_img", "mean"),
                          red=("bias_reduction", "mean"),
                          red_sd=("bias_reduction", "std"),
                          ici=("ici", "mean"))
                     .reset_index())
        self.gammas = np.sort(self.grid["gamma"].unique())
        self.deltas = np.sort(self.grid["delta"].unique())
        self.off = self.grid[(self.grid["gamma"] == 0) | (self.grid["delta"] == 0)]
        self.on = self.grid[(self.grid["gamma"] > 0) & (self.grid["delta"] > 0)]

    @staticmethod
    def _clean(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    @staticmethod
    def _panel_label(ax, letter, title, x=-0.14, y=1.045, offset=0.085):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=24,
                fontweight="bold", ha="left", va="bottom")
        ax.text(x + offset, y, title, transform=ax.transAxes, fontsize=17.5,
                ha="left", va="bottom")

    def _pivot(self, field):
        return self.cell.pivot(index="gamma", columns="delta", values=field).reindex(
            index=self.gammas, columns=self.deltas).values

    def _heatmap(self, ax, field, letter, title, cmap, vmin, vmax, outline=None, cbar_label=""):
        self._clean(ax)
        M = self._pivot(field)
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                shade = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                col = "white" if shade > 0.70 else "#222222"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=13.0, color=col)
        if outline is not None:
            gi = int(np.where(self.gammas == outline[0])[0][0])
            di = int(np.where(self.deltas == outline[1])[0][0])
            ax.add_patch(Rectangle((di - 0.5, gi - 0.5), 1, 1, fill=False,
                                   edgecolor=self.REF, linewidth=2.6))
        ax.set_xticks(range(len(self.deltas)))
        ax.set_xticklabels([f"{d:g}" for d in self.deltas])
        ax.set_yticks(range(len(self.gammas)))
        ax.set_yticklabels([f"{g:g}" for g in self.gammas])
        ax.set_xlabel(r"$\delta$, confounder into outcome")
        ax.set_ylabel(r"$\gamma$, into treatment")
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=12.5)
        cb.set_label(cbar_label, fontsize=13.5)
        self._panel_label(ax, letter, title, x=-0.20, offset=0.123)

    def _panel_a(self, ax):
        v = np.abs(np.concatenate([self._pivot("bias_s").ravel(), self._pivot("bias_i").ravel()])).max()
        self._heatmap(ax, "bias_s", "a", "Structured adjustment leaves bias",
                      "RdBu_r", -v, v, cbar_label="Bias (pp)")

    def _panel_b(self, ax):
        v = np.abs(np.concatenate([self._pivot("bias_s").ravel(), self._pivot("bias_i").ravel()])).max()
        self._heatmap(ax, "bias_i", "b", "Adding the image channel lowers it",
                      "RdBu_r", -v, v, cbar_label="Bias (pp)")

    def _panel_c(self, ax):
        M = self._pivot("red")
        self._heatmap(ax, "red", "c", "Bias reduction rises on both axes",
                      "Blues", float(M.min()), float(M.max()),
                      outline=(1.5, 1.5), cbar_label="Bias reduction (pp)")

    def _panel_d(self, ax):
        self._clean(ax)
        groups = [("Either channel off", self.off["bias_reduction"].values, self.NULLGRAY),
                  ("Both channels active", self.on["bias_reduction"].values, self.PRIMARY)]
        ax.axhspan(-self.AUC_THRESHOLD, self.AUC_THRESHOLD, color=self.REF, alpha=0.10, zorder=0)
        ax.axhline(0, color=self.REF, lw=1.1, ls="--", zorder=1)
        rng = np.random.default_rng(0)
        for k, (name, vals, col) in enumerate(groups):
            parts = ax.violinplot([vals], positions=[k], widths=0.72, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(col)
                b.set_alpha(0.30)
                b.set_edgecolor(col)
            ax.scatter(k + rng.normal(0, 0.055, len(vals)), vals, s=7, color=col,
                       alpha=0.30, linewidths=0, zorder=2)
            ax.hlines(vals.mean(), k - 0.36, k + 0.36, color=col, lw=3.2, zorder=4)
            if k == 0:
                continue
            ax.text(k - 0.40, vals.mean(), f"{vals.mean():.1f} (SD {vals.std(ddof=1):.1f})",
                    ha="right", va="center", fontsize=13.5, color=col, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.6))
        null = self.off["bias_reduction"].values
        t = stats.ttest_1samp(self.off["bias_reduction"], 0)
        lo, hi = np.percentile(self.off["bias_reduction"], [2.5, 97.5])
        ax.text(0.02, 0.97, f"null group: {null.mean():.1f} (SD {null.std(ddof=1):.1f})\n"
                            f"95% {lo:.1f} to {hi:.1f}\n"
                            f"$t$ = {t.statistic:.2f}, $p$ = {t.pvalue:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=13.5, color=self.REF)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.12)
        ax.set_xlim(-0.62, 1.48)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Either off\n$n$ = {len(self.off)}",
                            f"Both active\n$n$ = {len(self.on)}"])
        ax.set_ylabel("Bias reduction (pp)")
        self._panel_label(ax, "d", "Null where a channel is off", x=-0.20, offset=0.114)

    def _panel_e(self, ax):
        self._clean(ax)
        ax.scatter(self.grid["ici"], self.grid["bias_reduction"], s=11, color=self.PRIMARY,
                   alpha=0.18, linewidths=0, zorder=1)
        size = 40 + 260 * (self.cell["bias_s"] - self.cell["bias_s"].min()) / \
               (self.cell["bias_s"].max() - self.cell["bias_s"].min())
        ax.scatter(self.cell["ici"], self.cell["red"], s=size, color=self.PRIMARY,
                   edgecolor="white", linewidth=0.9, zorder=4)
        ax.axvline(self.AUC_THRESHOLD, color=self.REF, lw=1.3, ls="--", zorder=2)
        ax.axhline(0, color=self.REF, lw=1.0, ls=":", zorder=2)
        rc, pc = stats.spearmanr(self.cell["ici"], self.cell["red"])
        rr, pr = stats.spearmanr(self.grid["ici"], self.grid["bias_reduction"])
        ax.text(0.97, 0.05, f"cells $\\rho$ = {rc:.3f}\nruns $\\rho$ = {rr:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=14.0, color=self.REF)
        ax.text(self.AUC_THRESHOLD + 0.25, ax.get_ylim()[1] * 0.97, "threshold 2.0",
                fontsize=13.0, color=self.REF, ha="left", va="top", rotation=90)
        y0 = ax.get_ylim()[0]
        ax.scatter(self.real["ici"], np.full(len(self.real), y0 + 0.5), marker="|", s=210,
                   color=self.REAL, linewidths=2.0, zorder=5)
        ax.text(self.real["ici"].median(), y0 + 1.9, "real cells", fontsize=13.5,
                color=self.REAL, ha="center", va="bottom")
        ax.set_xlabel("Incremental Confounding Index (AUROC points)")
        ax.set_ylabel("Bias reduction (pp)")
        self._panel_label(ax, "e", "The index predicts bias reduction", x=-0.15, offset=0.075)

    def _panel_f(self, ax):
        self._clean(ax)
        z = self.grid[self.grid["ici"] <= 0]["bias_reduction"]
        lo, hi = np.percentile(z, [2.5, 97.5])
        ax.axvspan(lo, hi, color=self.NULLGRAY, alpha=0.18, zorder=0)
        ax.hist(z, bins=22, color=self.NULLGRAY, alpha=0.85, edgecolor="white", linewidth=0.7)
        ax.axvline(z.mean(), color=self.REF, lw=2.2, zorder=3)
        ax.axvline(self.AUC_THRESHOLD, color=self.REF, lw=1.3, ls="--", zorder=3)
        frac = 100.0 * (z > self.AUC_THRESHOLD).mean()
        ax.text(0.97, 0.95, f"$n$ = {len(z)} runs\nmean {z.mean():.1f} pp\n95% {lo:.1f} to {hi:.1f}\n"
                            f"{frac:.1f}% above 2 pp", transform=ax.transAxes,
                ha="right", va="top", fontsize=13.5, color=self.REF)
        ax.set_xlabel("Bias reduction (pp)")
        ax.set_ylabel("Runs")
        self._panel_label(ax, "f", "What an index at or below zero predicts", x=-0.16, offset=0.075)

    def _panel_g(self, ax):
        self._clean(ax)
        frac = (self.on.groupby(["gamma", "delta"])
                .apply(lambda x: 100.0 * x["bias_reduction"].mean() / x["bias_structured"].mean(),
                       include_groups=False)
                .sort_values())
        y = np.arange(len(frac))
        ax.hlines(y, 0, frac.values, color=self.PRIMARY, lw=2.0, alpha=0.75)
        ax.scatter(frac.values, y, s=95, color=self.PRIMARY, edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.axvline(np.median(frac.values), color=self.REF, lw=1.6, ls="--", zorder=2)
        ax.text(np.median(frac.values) + 1.5, -0.35,
                f"median {np.median(frac.values):.0f}%", fontsize=13.5,
                color=self.REF, ha="left", va="bottom")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{g:g}, {d:g}" for g, d in frac.index], fontsize=13.0)
        ax.set_xlim(0, max(frac.values) * 1.22)
        ax.set_xlabel("Structured bias removed (%)")
        ax.set_ylabel(r"cell ($\gamma$, $\delta$)")
        self._panel_label(ax, "g", "Recovery is substantial but partial", x=-0.20, offset=0.075)

    def _panel_h(self, ax):
        self._clean(ax)
        bg = self.grid.groupby("gamma")["positivity_cost"].agg(["mean", "std"])
        bd = self.grid.groupby("delta")["positivity_cost"].agg(["mean", "std"])
        ax.fill_between(bg.index, bg["mean"] - bg["std"], bg["mean"] + bg["std"],
                        color=self.PRIMARY, alpha=0.18, zorder=1)
        ax.plot(bg.index, bg["mean"], "-o", color=self.PRIMARY, lw=2.6, ms=10,
                mec="white", mew=1.0, zorder=3, label=r"against $\gamma$ (treatment)")
        ax.plot(bd.index, bd["mean"], "--s", color=self.NULLGRAY, lw=2.0, ms=8,
                mec="white", mew=0.9, zorder=3, label=r"against $\delta$ (outcome)")
        ax.axhline(self.COST_CAP, color=self.VETO, lw=1.6, ls="--", zorder=2)
        ax.text(bg.index[0], self.COST_CAP + 0.012, "cap 0.25", fontsize=13.5,
                color=self.VETO, ha="left", va="bottom")
        for x, v in [(bg.index[0], bg["mean"].iloc[0]), (bg.index[-1], bg["mean"].iloc[-1])]:
            ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(9, 10),
                        fontsize=13.0, color=self.PRIMARY)
        ax.legend(frameon=False, loc="upper left", fontsize=14.0, handletextpad=0.5)
        ax.set_xlabel("Coefficient value")
        ax.set_ylabel("Positivity cost")
        self._panel_label(ax, "h", "The cost tracks the treatment side", x=-0.114, offset=0.047)

    def _panel_i(self, ax):
        self._clean(ax)
        g = self.grid
        veto = g[(g["ici"] >= self.AUC_THRESHOLD) & (g["positivity_cost"] > self.COST_CAP)]
        acc = g[(g["ici"] >= self.AUC_THRESHOLD) & (g["positivity_cost"] <= self.COST_CAP)]
        dec = g[g["ici"] < self.AUC_THRESHOLD]
        for sub, col, lab in [(dec, self.NULLGRAY, "declined on the index"),
                              (acc, self.PRIMARY, "accepted"),
                              (veto, self.VETO, "vetoed by the cap")]:
            ax.scatter(sub["ici"], sub["positivity_cost"], s=15, color=col, alpha=0.55,
                       linewidths=0, zorder=2, label=lab)
        ax.axvline(self.AUC_THRESHOLD, color=self.REF, lw=1.3, ls="--", zorder=3)
        ax.axhline(self.COST_CAP, color=self.REF, lw=1.3, ls="--", zorder=3)
        ax.text(0.985, 0.60, f"vetoed: $n$ = {len(veto)}\nmean {veto['bias_reduction'].mean():.1f} pp\n"
                             "of bias reduction", transform=ax.transAxes, ha="right", va="top",
                fontsize=13.5, color=self.VETO,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.5))
        ax.set_xlabel("Incremental Confounding Index (AUROC points)")
        ax.set_ylabel("Positivity cost")
        self._panel_label(ax, "i", "The cap can veto a useful modality", x=-0.114, offset=0.047)

    def _legend(self, ax):
        ax.axis("off")
        handles = [
            Line2D([0], [0], marker="o", lw=0, ms=14, mfc=self.PRIMARY, mec="white",
                   label="Both confounding channels active"),
            Line2D([0], [0], marker="o", lw=0, ms=14, mfc=self.NULLGRAY, mec="white",
                   label="Either channel switched off"),
            Line2D([0], [0], marker="|", lw=0, ms=18, mec=self.REAL, mew=2.5,
                   label="Real modality-by-emulation cells"),
            Line2D([0], [0], marker="o", lw=0, ms=14, mfc=self.VETO, mec="white",
                   label="Vetoed by the positivity cap"),
            Line2D([0], [0], ls="--", lw=1.6, color=self.REF, label="Decision threshold"),
        ]
        ax.legend(handles=handles, ncol=3, frameon=False, loc="center",
                  columnspacing=1.9, handletextpad=0.55, fontsize=16.0)

    def build(self):
        self.fig = plt.figure(figsize=(19.0, 20.5), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.995, bottom=0.947)
        gs_r1 = self.fig.add_gridspec(1, 4, left=0.055, right=0.985, top=0.912, bottom=0.672,
                                      wspace=0.52)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.600, bottom=0.352,
                                      wspace=0.34)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.075, right=0.965, top=0.278, bottom=0.045,
                                      wspace=0.22)
        self._legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_a(self.fig.add_subplot(gs_r1[0, 0]))
        self._panel_b(self.fig.add_subplot(gs_r1[0, 1]))
        self._panel_c(self.fig.add_subplot(gs_r1[0, 2]))
        self._panel_d(self.fig.add_subplot(gs_r1[0, 3]))
        self._panel_e(self.fig.add_subplot(gs_r2[0, 0]))
        self._panel_f(self.fig.add_subplot(gs_r2[0, 1]))
        self._panel_g(self.fig.add_subplot(gs_r2[0, 2]))
        self._panel_h(self.fig.add_subplot(gs_r3[0, 0]))
        self._panel_i(self.fig.add_subplot(gs_r3[0, 1]))
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
    Figure2Calibration().build().save()
