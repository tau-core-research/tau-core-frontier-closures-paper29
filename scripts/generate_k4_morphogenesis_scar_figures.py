#!/usr/bin/env python3
"""
Generate Publication-Grade Bilingual (EN & HU) 2x2 Vector PDF & 300-DPI PNG Figures
for the K4 Morphogenesis Scar ("Stage Observable") Audit (Paper 29 v3.4.0 / Module G).

Reads verified Stage-A JSON:
  ../data/k4_morphogenesis_scar_results.json (or hub fallback)
Writes:
  ../figures/fig_k4_morphogenesis_scar_cmb_bounce_en.{pdf,png}
  ../figures/fig_k4_morphogenesis_scar_cmb_bounce_hu.{pdf,png}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results() -> dict:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "data" / "k4_morphogenesis_scar_results.json",
        here.parents[2] / "tau-core-theory" / "k4_morphogenesis_scar_audit_2026_11_26" / "k4_morphogenesis_scar_results.json",
    ]
    for p in candidates:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("Could not find k4_morphogenesis_scar_results.json")


def make_figure(res: dict, lang: str, out_dir: Path) -> None:
    is_hu = (lang == "hu")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.9,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), dpi=300)
    (ax1, ax2), (ax3, ax4) = axes

    # Palette (colorblind-safe, clean journal style)
    c_blue = "#1f5a99"
    c_orange = "#d95f02"
    c_teal = "#1b9e77"
    c_red = "#b2182b"
    c_purple = "#7570b3"
    c_gray = "#555555"

    # ------------------------------------------------------------------
    # PANEL (a): Low-ell Morphogenesis Scar Profile (ell = 2..30)
    # ------------------------------------------------------------------
    prof = res["low_ell_profile_at_ell0_5.086"]
    ells = np.array([p["ell"] for p in prof])
    r_sw = np.array([p["r_TT_pure_SW"] for p in prof])
    r_isw = np.array([p["r_TT_with_ISW25"] for p in prof])
    r_reion = np.array([p["r_reion_shell_EE_BB_proxy"] for p in prof])

    ax1.axhline(1.0, color=c_gray, lw=1.0, ls=":", label="Bunch–Davies (1,00)" if is_hu else "Bunch–Davies (1.00)")
    ax1.plot(ells, r_sw, "o-", color=c_blue, lw=2.0, ms=4.5,
             label="TT tiszta SW ($\\ell_0=5{,}09$)" if is_hu else "TT pure SW ($\\ell_0=5.09$)")
    ax1.plot(ells, r_isw, "s--", color=c_orange, lw=1.7, ms=4.0,
             label="TT + 25% kései ISW" if is_hu else "TT + 25% late-time ISW")
    ax1.plot(ells, r_reion, "^-.", color=c_teal, lw=1.8, ms=4.5,
             label="EE/BB reionizációs proxy ($\\chi_{\\mathrm{reion}}=0{,}65\\chi_{\\mathrm{LSS}}$)" if is_hu
             else "EE/BB reion.-shell proxy ($\\chi_{\\mathrm{reion}}=0.65\\chi_{\\mathrm{LSS}}$)")
    ax1.scatter([2], [0.222], s=75, color=c_red, zorder=6, marker="*",
                label="Planck 2018 kvadrupól ($r_2\\approx 0{,}222$)" if is_hu
                else "Planck 2018 quadrupole ($r_2\\approx 0.222$)")

    ax1.annotate("+11,4% (TT, $\\ell=11$)" if is_hu else "+11.4% (TT, $\\ell=11$)",
                 xy=(11, 1.1136), xytext=(13.5, 1.21),
                 arrowprops=dict(arrowstyle="->", color=c_blue, lw=0.9),
                 fontsize=8.0, color=c_blue, fontweight="bold")
    ax1.annotate("+10,8% (EE/BB, $\\ell=7$)" if is_hu else "+10.8% (EE/BB, $\\ell=7$)",
                 xy=(7, 1.108), xytext=(3.2, 1.23),
                 arrowprops=dict(arrowstyle="->", color=c_teal, lw=0.9),
                 fontsize=8.0, color=c_teal, fontweight="bold")

    ax1.set_xlim(1.5, 31)
    ax1.set_ylim(0.08, 1.32)
    ax1.set_xlabel("Multipólus $\\ell$" if is_hu else "Multipole $\\ell$")
    ax1.set_ylabel("Moduláció $C_\\ell / C_\\ell^{\\mathrm{BD}}$" if is_hu else "Modulation $C_\\ell / C_\\ell^{\\mathrm{BD}}$")
    ax1.set_title("(a) Alacsony-$\\ell$ csíraevolúciós hegprofil ($N_{\\mathrm{tot}}=56{,}33$)" if is_hu
                  else "(a) Low-$\\ell$ Morphogenesis Scar Profile ($N_{\\mathrm{tot}}=56.33$)")
    ax1.grid(True)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92)

    # ------------------------------------------------------------------
    # PANEL (b): Official Planck 2018 PR3 TT (83 bins) Exclusion Curve
    # ------------------------------------------------------------------
    scan = res["planck_scan_curve"]
    l0_s = np.array([r["ell_0"] for r in scan])
    d_bes_kin = np.array([r["dchi2_TT_kin_Bessel"] for r in scan])
    d_lim_kin = np.array([r["dchi2_TT_kin_Limber"] for r in scan])
    d_bes_mink = np.array([r["dchi2_TT_mink_Bessel"] for r in scan])

    ax2.axvspan(20.0, 30.5, color=c_red, alpha=0.10,
                label="Planck TT kizárt ($\\ell_0\\geq 20$, 95% CL)" if is_hu
                else "Planck TT excluded ($\\ell_0\\geq 20$, 95% CL)")
    ax2.axvspan(4.0, 10.31, color=c_teal, alpha=0.12,
                label="Fennmaradó kvadrupól-ablak ($\\ell_0\\in[4;\\,10{,}3]$)" if is_hu
                else "Surviving quadrupole window ($\\ell_0\\in[4,\\,10.3]$)")

    ax2.plot(l0_s, d_bes_kin, "o-", color=c_blue, lw=2.0, ms=3.5,
             label="Kination $\\to$ dS (Bessel, elsődleges)" if is_hu else "Kination $\\to$ dS (Bessel, primary)")
    ax2.plot(l0_s, d_lim_kin, "--", color=c_orange, lw=1.5,
             label="Kination $\\to$ dS (Limber-diagnosztika)" if is_hu else "Kination $\\to$ dS (Limber diagnostic)")
    ax2.plot(l0_s, d_bes_mink, "-.", color=c_purple, lw=1.7,
             label="Minkowski $\\to$ dS (Bessel)" if is_hu else "Minkowski $\\to$ dS (Bessel)")

    ax2.axhline(3.841, color=c_red, ls=":", lw=1.1)
    ax2.axhline(9.0, color=c_red, ls="--", lw=1.0)
    ax2.text(1.5, 4.3, "95% CL ($\\Delta\\chi^2=3{,}84$)" if is_hu else "95% CL ($\\Delta\\chi^2=3.84$)",
             color=c_red, fontsize=7.8)
    ax2.text(1.5, 9.6, "$3\\sigma$ ($\\Delta\\chi^2=9{,}00$)" if is_hu else "$3\\sigma$ ($\\Delta\\chi^2=9.00$)",
             color=c_red, fontsize=7.8)

    ax2.set_xlim(1.0, 26.5)
    ax2.set_ylim(-0.8, 22.0)
    ax2.set_xlabel("Átmeneti skála $\\ell_0 = k_0 \\chi_{\\mathrm{LSS}}$" if is_hu
                   else "Transition scale $\\ell_0 = k_0 \\chi_{\\mathrm{LSS}}$")
    ax2.set_ylabel("Planck 2018 TT (83 sáv) $\\Delta\\chi^2$" if is_hu
                   else "Planck 2018 TT (83 bins) $\\Delta\\chi^2$")
    ax2.set_title("(b) Planck 2018 PR3 TT kizárási határ ($\\ell\\geq 47{,}7$)" if is_hu
                  else "(b) Planck 2018 PR3 TT Exclusion Bound ($\\ell\\geq 47.7$)")
    ax2.grid(True)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.5)

    # Top x-axis for N_tot e-folds
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    tick_l0 = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    tick_ntot = [51.412 + math.log(0.05 / (l0 / 13870.0)) for l0 in tick_l0]
    ax2_top.set_xticks(tick_l0)
    ax2_top.set_xticklabels([f"{n:.1f}".replace(".", ",") if is_hu else f"{n:.1f}" for n in tick_ntot], fontsize=8.0)
    ax2_top.set_xlabel("Teljes infláció $N_{\\mathrm{tot}}$ [e-fold] (Paper 24)" if is_hu
                       else "Total inflation $N_{\\mathrm{tot}}$ [e-folds] (Paper 24)", fontsize=8.5)

    # ------------------------------------------------------------------
    # PANEL (c): Wishart Cosmic-Variance Discovery SNR (ell = 2..29, f_sky = 0.70)
    # ------------------------------------------------------------------
    wf = res["wishart_low_ell_forecast"]
    l0_w = np.array([w["ell_0"] for w in wf])
    snr_tt = np.array([w["SNR_disc_TT_fsky070"] for w in wf])
    snr_isw = np.array([w["SNR_disc_TT_ISW25_fsky070"] for w in wf])
    snr_ttee = np.array([w["SNR_disc_TT_plus_EEproxy_fsky070"] for w in wf])

    ax3.plot(l0_w, snr_tt, "o-", color=c_blue, lw=2.0, ms=5.0,
             label="TT tiszta SW ($f_{\\mathrm{sky}}=0{,}70$)" if is_hu else "TT pure SW ($f_{\\mathrm{sky}}=0.70$)")
    ax3.plot(l0_w, snr_isw, "s--", color=c_orange, lw=1.8, ms=4.5,
             label="TT + 25% ISW ($f_{\\mathrm{sky}}=0{,}70$)" if is_hu else "TT + 25% ISW ($f_{\\mathrm{sky}}=0.70$)")
    ax3.plot(l0_w, snr_ttee, "^-.", color=c_teal, lw=2.0, ms=5.0,
             label="TT + EE reion.-proxy ($f_{\\mathrm{sky}}=0{,}70$)" if is_hu
             else "TT + EE reion.-proxy ($f_{\\mathrm{sky}}=0.70$)")

    for sig, col in [(2.0, c_gray), (3.0, c_orange), (5.0, c_red)]:
        ax3.axhline(sig, color=col, ls=":", lw=1.0)
        ax3.text(14.1, sig + 0.12, f"${int(sig)}\\sigma$", color=col, fontsize=8.0, ha="right")

    ax3.axvline(5.086, color=c_blue, ls="--", lw=0.9, alpha=0.6)
    ax3.text(5.25, 0.7, "$\\ell_0^{\\mathrm{SW}}=5{,}09$" if is_hu else "$\\ell_0^{\\mathrm{SW}}=5.09$",
             fontsize=8.0, color=c_blue)

    ax3.set_xlim(1.5, 14.5)
    ax3.set_ylim(0.0, 11.2)
    ax3.set_xlabel("Átmeneti skála $\\ell_0$" if is_hu else "Transition scale $\\ell_0$")
    ax3.set_ylabel("Várható Wishart SNR [$\\sigma$] ($\\ell=2\\dots 29$)" if is_hu
                   else "Expected Wishart SNR [$\\sigma$] ($\\ell=2\\dots 29$)")
    ax3.set_title("(c) Alacsony-$\\ell$ Wishart felfedezési SNR ($f_{\\mathrm{sky}}=0{,}70$)" if is_hu
                  else "(c) Low-$\\ell$ Wishart Discovery SNR ($f_{\\mathrm{sky}}=0.70$)")
    ax3.grid(True)
    ax3.legend(loc="upper left", frameon=True, framealpha=0.92)

    # ------------------------------------------------------------------
    # PANEL (d): Two Proven No-Go Theorems (Single-Field Kination & P26 Axion)
    # ------------------------------------------------------------------
    ng_kin = res["candidate2_subplaquette_single_field_kination_nogo"]
    phi_dom = ng_kin["phi_E_subplaquette_domain_MPl"][1]   # 8.2514
    phi_req = ng_kin["Delta_phi_E_kination_MPl"]           # 9.306
    phi_scar = ng_kin["phi_E_0_at_Ntot_56.33_MPl"]         # 5.3803
    phi_down = phi_scar + phi_req                          # 14.686

    labels_d = (
        ["$\\phi_E^{(0)}$\n($N_{\\mathrm{tot}}=56{,}3$)",
         "Szub-plakett\nhatár ($\\Phi=M_*$)",
         "Kination út\n$\\Delta\\phi_E^{\\mathrm{kin}}$",
         "1-mezős indulás\n($\\Phi_{\\mathrm{b}}=13{,}8\\,M_*$)"]
        if is_hu else
        ["$\\phi_E^{(0)}$\n($N_{\\mathrm{tot}}=56.3$)",
         "Sub-plaquette\nceiling ($\\Phi=M_*$)",
         "Kination roll\n$\\Delta\\phi_E^{\\mathrm{kin}}$",
         "1-field start\n($\\Phi_{\\mathrm{b}}=13.8\\,M_*$)"]
    )
    vals_d = [phi_scar, phi_dom, phi_req, phi_down]
    cols_d = [c_teal, c_blue, c_orange, c_red]

    bars = ax4.bar(range(4), vals_d, color=cols_d, alpha=0.88, width=0.58, edgecolor="#222222", lw=0.8)
    ax4.axhline(phi_dom, color=c_blue, ls="--", lw=1.2)
    for bar, v in zip(bars, vals_d):
        txt = f"{v:.2f}".replace(".", ",") if is_hu else f"{v:.2f}"
        ax4.text(bar.get_x() + bar.get_width() / 2.0, v + 0.35, f"{txt} $M_{{\\mathrm{{Pl}}}}$",
                 ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    box_txt = (
        "1. PROVEN NO-GO (1-mezős kination):\n"
        "   $\\Delta\\phi_E^{\\mathrm{kin}}=9{,}31 > 8{,}25\\,M_{\\mathrm{Pl}}$ ($\\Phi\\leq M_*$)\n"
        "   $\\Rightarrow$ extra merev módus (pl. Paper 28 $\\psi_{\\mathrm{kin}}$) kell!\n"
        "2. PROVEN NO-GO (Paper 26 $Z_{16}$ axion):\n"
        "   $\\beta_{\\max}=2{,}37\\times 10^{-17}\\,^\\circ \\ll 0{,}30^\\circ$ (CMB EB)"
        if is_hu else
        "1. PROVEN NO-GO (1-field kination):\n"
        "   $\\Delta\\phi_E^{\\mathrm{kin}}=9.31 > 8.25\\,M_{\\mathrm{Pl}}$ ($\\Phi\\leq M_*$)\n"
        "   $\\Rightarrow$ extra stiff DOF (e.g. Paper 28 $\\psi_{\\mathrm{kin}}$) required!\n"
        "2. PROVEN NO-GO (Paper 26 $Z_{16}$ axion):\n"
        "   $\\beta_{\\max}=2.37\\times 10^{-17}\\,^\\circ \\ll 0.30^\\circ$ (CMB EB)"
    )
    ax4.text(0.03, 0.96, box_txt, transform=ax4.transAxes, va="top", ha="left",
             fontsize=7.6, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#888888", lw=0.8))

    ax4.set_xticks(range(4))
    ax4.set_xticklabels(labels_d, fontsize=8.0)
    ax4.set_ylim(0, 19.5)
    ax4.set_ylabel("Kanonikus mezőérték $\\phi_E / M_{\\mathrm{Pl}}$" if is_hu
                   else "Canonical field excursion $\\phi_E / M_{\\mathrm{Pl}}$")
    ax4.set_title("(d) Big Bounce $\\to$ Starobinsky & Axion NO-GO határok" if is_hu
                  else "(d) Big Bounce $\\to$ Starobinsky & Axion NO-GO Bounds")
    ax4.grid(True, axis="y")

    suptitle = (
        "Tau Core $K_4$ csíraevolúciós heg-audit (Paper 29 v3.4.0 / Modul G · Stage-A referens-auditált · NATURE_LEVEL_CLOSURE = false)"
        if is_hu else
        "Tau Core $K_4$ Morphogenesis Scar Audit (Paper 29 v3.4.0 / Module G · Stage-A Referee-Audited · NATURE_LEVEL_CLOSURE = false)"
    )
    fig.suptitle(suptitle, fontsize=11.0, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.962])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig_k4_morphogenesis_scar_cmb_bounce_{lang}"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    res = load_results()
    here = Path(__file__).resolve()
    fig_dir = here.parents[1] / "figures"
    hub_dir = here.parents[2] / "tau-core-theory" / "k4_morphogenesis_scar_audit_2026_11_26"
    for lang in ("en", "hu"):
        make_figure(res, lang, fig_dir)
        if hub_dir.is_dir():
            make_figure(res, lang, hub_dir)
    print("Generated EN & HU publication-grade PDF + 300-DPI PNG figures cleanly.")


if __name__ == "__main__":
    main()
