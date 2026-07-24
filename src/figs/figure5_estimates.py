"""
figs/figure5_estimates.py
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
from matplotlib.patches import Patch


class Figure5Estimates:
    EFFECTS_CSV = "inputs/results/effects.csv"
    CONTRASTS_CSV = "inputs/results/contrasts.csv"
    OUT_PDF = "figs/figure5_estimates.pdf"

    COND_ORDER = ["naive", "expert", "structured", "struct_img", "struct_radtext",
                  "struct_histnote", "multimodal"]
    COND_LABELS = {"naive": "Unadjusted", "expert": "Expert set", "structured": "Structured",
                   "struct_img": "$+$ radiograph", "struct_radtext": "$+$ report",
                   "struct_histnote": "$+$ summary", "multimodal": "$+$ all three"}
    COND_COLORS = {"naive": "#BBBBBB", "expert": "#777777", "structured": "#333333",
                   "struct_img": "#2166AC", "struct_radtext": "#D6604D",
                   "struct_histnote": "#E08214", "multimodal": "#762A83"}
    MOD_COLORS = {"images": "#2166AC", "radtext": "#D6604D", "histnote": "#E08214",
                  "multimodal": "#762A83"}
    MOD_LABELS = {"images": "radiograph", "radtext": "report",
                  "histnote": "summary", "multimodal": "all three"}
    EMU_ORDER = ["fluids_sepsis", "transfusion_threshold", "rrt_timing", "prone_positioning"]
    EMU_LABELS = {"fluids_sepsis": "Fluid strategy", "transfusion_threshold": "Transfusion",
                  "rrt_timing": "RRT timing", "prone_positioning": "Proning"}
    EMU_MARKERS = {"fluids_sepsis": "o", "transfusion_threshold": "s",
                   "rrt_timing": "^", "prone_positioning": "D"}
    REF = "#333333"
    TRIAL = "#1B7837"

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
        e = pd.read_csv(self.EFFECTS_CSV)
        self.p = e[(e["dataset"] == "mimic") & (e["cohort"] == "imaged")].copy()
        c = pd.read_csv(self.CONTRASTS_CSV)
        self.bias = c[c["contrast"] == "bias_reduction"].copy()
        self.comp = c[c["contrast"] == "complementary_bias_reduction"].copy()
        self.expert = c[c["contrast"] == "multimodal_vs_expert"].copy()
        self.absbias = (self.p.pivot_table(index="intervention", columns="condition",
                                           values="bias_point").abs()
                        .reindex(index=self.EMU_ORDER, columns=self.COND_ORDER))

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

    def _forest(self, ax, emu, letter, title, ylabels=True):
        self._clean(ax)
        sub = self.p[self.p["intervention"] == emu].set_index("condition")
        ref = sub.iloc[0]
        ax.axvspan(ref["ref_ci_low"], ref["ref_ci_high"], color=self.TRIAL, alpha=0.14, zorder=0)
        ax.axvline(ref["ref_rd"], color=self.TRIAL, lw=1.8, ls="-", zorder=1)
        for k, cond in enumerate(self.COND_ORDER):
            row = sub.loc[cond]
            y = len(self.COND_ORDER) - 1 - k
            if bool(row["undefined"]) or pd.isna(row["effect_point"]):
                ax.text(0.5, y, "not estimable", transform=ax.get_yaxis_transform(),
                        ha="center", va="center", fontsize=13.5, color=self.REF, style="italic")
                continue
            col = self.COND_COLORS[cond]
            ax.errorbar(row["effect_point"], y,
                        xerr=[[row["effect_point"] - row["effect_ci_low"]],
                              [row["effect_ci_high"] - row["effect_point"]]],
                        fmt="none", ecolor=col, elinewidth=2.0, capsize=4, zorder=3)
            ax.scatter(row["effect_point"], y, s=125, color=col, edgecolor="white",
                       linewidth=1.0, zorder=4)
        ax.set_yticks(range(len(self.COND_ORDER)))
        ax.set_yticklabels([self.COND_LABELS[c] for c in reversed(self.COND_ORDER)]
                           if ylabels else [])
        ax.set_ylim(-0.7, len(self.COND_ORDER) - 0.3)
        ax.set_xlabel("Risk difference (pp)")
        self._panel_label(ax, letter, title, x=-0.2515 if ylabels else -0.14,
                          offset=0.063 if ylabels else 0.082)

    def _panel_a(self, ax):
        self._forest(ax, "fluids_sepsis", "a", "Fluid strategy in sepsis")

    def _panel_b(self, ax):
        self._forest(ax, "transfusion_threshold", "b", "Transfusion threshold", ylabels=False)

    def _panel_c(self, ax):
        self._forest(ax, "rrt_timing", "c", "RRT timing in AKI", ylabels=False)

    def _panel_d(self, ax):
        """Distance from the trial reference: the three baseline rungs as points, and the
        four modality conditions as a range, per emulation."""
        self._clean(ax)
        base = ["naive", "expert", "structured"]
        mods = ["struct_img", "struct_radtext", "struct_histnote", "multimodal"]
        x = np.arange(len(self.EMU_ORDER))
        for k, emu in enumerate(self.EMU_ORDER):
            v = self.absbias.loc[emu]
            for j, cond in enumerate(base):
                if np.isnan(v[cond]):
                    continue
                ax.scatter(k - 0.22 + j * 0.15, v[cond], s=135, marker="o",
                           color=self.COND_COLORS[cond], edgecolor="white",
                           linewidth=1.0, zorder=4)
            mv = v[mods].values.astype(float)
            mv = mv[~np.isnan(mv)]
            if len(mv):
                ax.vlines(k + 0.30, mv.min(), mv.max(), color=self.MOD_COLORS["multimodal"],
                          lw=7.0, alpha=0.35, zorder=2)
                ax.scatter(np.full(len(mv), k + 0.30), mv, s=70, color="white",
                           edgecolor=self.MOD_COLORS["multimodal"], linewidth=1.8, zorder=4)
            else:
                ax.text(k + 0.30, 2.0, "not\nestimable", ha="center", va="center",
                        fontsize=12.5, color=self.REF, style="italic")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([self.EMU_LABELS[e].replace(" ", "\n") for e in self.EMU_ORDER],
                           fontsize=13.5)
        ax.set_xlim(-0.62, len(self.EMU_ORDER) - 0.35)
        ax.set_ylabel("Distance from trial (pp, log)")
        self._panel_label(ax, "d", "The structured step does the work",
                          x=-0.225, offset=0.100)

    def _contrast_panel(self, ax, df, letter, title, labeller, colorer, ylabels=True,
                        label_x=-0.317, label_offset=0.077):
        self._clean(ax)
        s = df.sort_values("value_point").reset_index(drop=True)
        y = np.arange(len(s))
        for k, row in s.iterrows():
            col = colorer(row)
            ax.errorbar(row["value_point"], k,
                        xerr=[[row["value_point"] - row["value_ci_low"]],
                              [row["value_ci_high"] - row["value_point"]]],
                        fmt="none", ecolor=col, elinewidth=1.8, capsize=3, zorder=3)
            ax.scatter(row["value_point"], k, s=90, color=col, edgecolor="white",
                       linewidth=0.8, zorder=4)
        ax.axvline(0, color=self.REF, lw=1.5, ls="--", zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([labeller(r) for _, r in s.iterrows()] if ylabels else [],
                           fontsize=12.5)
        ax.set_ylim(-0.7, len(s) + 0.45)
        n_span = int(((s["value_ci_low"] <= 0) & (s["value_ci_high"] >= 0)).sum())
        ax.text(0.97, 0.97, f"all {n_span} of {len(s)} span zero", transform=ax.transAxes,
                ha="right", va="top", fontsize=14.0, color=self.REF, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2.0))
        ax.set_xlabel("Paired change in distance (pp)")
        self._panel_label(ax, letter, title, x=label_x, offset=label_offset)

    def _panel_e(self, ax):
        est = self.bias[self.bias["intervention"] != "prone_positioning"]
        self._contrast_panel(
            ax, est, "e", "No modality moves the estimate",
            lambda r: f"{self.EMU_LABELS[r['intervention']]}, {self.MOD_LABELS[r['modality']]}",
            lambda r: self.MOD_COLORS[r["modality"]],
            label_x=-0.317, label_offset=0.077)

    def _panel_f(self, ax):
        self._clean(ax)
        g = self.bias.groupby("intervention")["min_detectable_pp"].agg(["min", "max"])
        g = g.reindex(self.EMU_ORDER)
        y = np.arange(len(g))
        for k, emu in enumerate(self.EMU_ORDER):
            ax.plot([g.loc[emu, "min"], g.loc[emu, "max"]], [k, k], color=self.REF,
                    lw=3.0, alpha=0.55, zorder=2, solid_capstyle="round")
            for v in (g.loc[emu, "min"], g.loc[emu, "max"]):
                ax.scatter(v, k, s=100, marker=self.EMU_MARKERS[emu], color=self.REF,
                           edgecolor="white", linewidth=0.9, zorder=4)
            ax.text(g.loc[emu, "max"] * 1.20, k, f"{g.loc[emu, 'min']:.1f} to {g.loc[emu, 'max']:.1f}",
                    va="center", fontsize=13.5, color=self.REF)
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels([self.EMU_LABELS[e] for e in self.EMU_ORDER])
        ax.set_ylim(-0.7, len(g) - 0.3)
        ax.set_xlim(1.2, 4200)
        ax.set_xlabel("Min. detectable effect (pp, log)")
        self._panel_label(ax, "f", "Only the fluid null is decisive", x=-0.433, offset=0.105)

    def _panel_g(self, ax):
        est = self.comp[self.comp["intervention"] != "prone_positioning"]
        self._contrast_panel(
            ax, est, "g", "No channel contributes only jointly",
            lambda r: f"{self.EMU_LABELS[r['intervention']]}, {self.MOD_LABELS[r['modality']]}",
            lambda r: self.MOD_COLORS[r["modality"]],
            label_x=-0.342, label_offset=0.052)

    def _panel_h(self, ax):
        self._clean(ax)
        s = self.expert.sort_values("value_point").reset_index(drop=True)
        sizes = (self.p[self.p["condition"] == "expert"]
                 .set_index("intervention")["expert_confounders_extracted"])
        for k, row in s.iterrows():
            col = self.MOD_COLORS["multimodal"]
            ax.errorbar(row["value_point"], k,
                        xerr=[[row["value_point"] - row["value_ci_low"]],
                              [row["value_ci_high"] - row["value_point"]]],
                        fmt="none", ecolor=col, elinewidth=1.8, capsize=3, zorder=3)
            ax.scatter(row["value_point"], k, s=110,
                       marker=self.EMU_MARKERS[row["intervention"]], color=col,
                       edgecolor="white", linewidth=0.9, zorder=4)
        ax.axvline(0, color=self.REF, lw=1.5, ls="--", zorder=2)
        est = s[s["intervention"] != "prone_positioning"]
        lim = max(abs(est["value_ci_low"].min()), abs(est["value_ci_high"].max())) * 1.40
        ax.set_xlim(-lim, lim)
        for k, row in s.iterrows():
            if row["intervention"] == "prone_positioning":
                ax.annotate(f"interval {row['value_ci_low']:.0f} to {row['value_ci_high']:.0f}",
                            (0, k), textcoords="offset points", xytext=(0, 15), ha="center",
                            fontsize=13.0, color=self.MOD_COLORS["multimodal"])
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([f"{self.EMU_LABELS[r['intervention']]} "
                            f"({int(sizes[r['intervention']])} variables)"
                            for _, r in s.iterrows()], fontsize=13.0)
        ax.set_ylim(-0.7, len(s) - 0.3)
        ax.set_xlabel("Paired change in distance (pp)")
        ax.text(0.97, 0.04, f"all {len(s)} of {len(s)} span zero", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=14.0, color=self.REF)
        self._panel_label(ax, "h", "The multimodal set does not beat expert curation",
                          x=-0.475, offset=0.065)

    def _legend(self, ax):
        ax.axis("off")
        conds = [Line2D([0], [0], marker="o", lw=0, ms=13, mfc=self.COND_COLORS[c],
                        mec="white", label=self.COND_LABELS[c]) for c in self.COND_ORDER]
        extra = [Patch(facecolor=self.TRIAL, alpha=0.25, label="Randomized trial reference"),
                 Line2D([0], [0], marker="o", lw=7.0, ms=10, color=self.MOD_COLORS["multimodal"],
                        alpha=0.55, mfc="white", mec=self.MOD_COLORS["multimodal"], mew=1.8,
                        label="Range over the four modality conditions")]
        ax.legend(handles=conds + extra, ncol=4, frameon=False, loc="center",
                  columnspacing=1.7, handletextpad=0.55, fontsize=15.0)

    def build(self):
        self.fig = plt.figure(figsize=(19.0, 20.0), facecolor="white")
        gs_leg = self.fig.add_gridspec(1, 1, left=0.045, right=0.985, top=0.995, bottom=0.940)
        gs_r1 = self.fig.add_gridspec(1, 3, left=0.105, right=0.980, top=0.895, bottom=0.660,
                                      width_ratios=[1.30, 1.0, 1.0], wspace=0.22)
        gs_r2 = self.fig.add_gridspec(1, 3, left=0.105, right=0.980, top=0.588, bottom=0.330,
                                      width_ratios=[1.0, 1.30, 0.95], wspace=0.62)
        gs_r3 = self.fig.add_gridspec(1, 2, left=0.154, right=0.980, top=0.258, bottom=0.045,
                                      width_ratios=[1.25, 1.0], wspace=0.52)
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
    Figure5Estimates().build().save()
