#!/usr/bin/env python3
"""
JSON-Driven Vector PDF Figure Generator (EN + HU) for Tau Core Paper 29 — Step 3 (Pillar 3, v2.0.0):
  - figures/fig5_24cell_isotropy_and_s4_planck_shield[_hu].pdf
  - figures/fig6_frullani_rg_simplicial_gamma_and_void_homogenization[_hu].pdf

Strict Referee Formatting Rules:
  - Must pass `python3 -W error` cleanly with exit code 0.
  - 100% JSON-driven curves and metrics (zero duplicate physics solvers).
  - 100% Hungarian decimal commas (`{,}` inside mathtext, `,` in plain text) and zero English leaks on `_hu.pdf`.
  - Uses `\\leq` / `\\geq` (never `\\le` / `\\ge`) for Matplotlib mathtext compatibility.
  - Wide subplot gutters (`wspace=0.40, hspace=0.38`) and ample headroom to guarantee zero overlaps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
JSON_IN = ROOT_DIR / "paper29_step3_verification_results.json"
FIG_OUT = ROOT_DIR / "figures"


def fmt_num(val: float, decimals: int, lang: str, math_mode: bool = False) -> str:
    s = f"{val:.{decimals}f}"
    if lang == "hu":
        return s.replace(".", "{,}" if math_mode else ",")
    return s


def fmt_sci(val: float, decimals: int, lang: str) -> str:
    if val == 0.0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    mant = val / (10.0**exp)
    m_str = f"{mant:.{decimals}f}"
    if lang == "hu":
        m_str = m_str.replace(".", "{,}")
    return rf"{m_str}\times 10^{{{exp}}}"


def make_comma_formatter(decimals: int = 1) -> FuncFormatter:
    def _fmt(x: float, _pos: int) -> str:
        if abs(x - round(x)) < 1.0e-9 and decimals == 0:
            return f"{int(round(x))}"
        return f"{x:.{decimals}f}".replace(".", ",")
    return FuncFormatter(_fmt)


def make_log_comma_formatter() -> FuncFormatter:
    def _fmt(x: float, _pos: int) -> str:
        if x <= 0:
            return "0"
        lg = math.log10(x)
        if abs(lg - round(lg)) < 1.0e-6:
            return rf"$10^{{{int(round(lg))}}}$"
        return ""
    return FuncFormatter(_fmt)


def generate_fig5(res: Dict[str, Any], lang: str) -> Path:
    g1 = res["gate_S3_1_24cell_isotropy_and_os_positivity"]
    g2 = res["gate_S3_2_k4_defect_algebra_and_continuum_couplings"]
    inv = g1["tensor_ring_invariants_r1_to_6"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 9.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.3))
    fig.subplots_adjust(left=0.125, right=0.978, top=0.935, bottom=0.085, wspace=0.38, hspace=0.38)

    # --- Panel A: W(B_4) vs W(F_4) vs O(4) Tensor-Ring Invariants r = 1..6 ---
    ax = axes[0, 0]
    ranks = np.arange(1, 7)
    d_wb4 = [inv[f"rank_{r}"]["W_B4_hypercubic_384"] for r in ranks]
    d_wf4 = [inv[f"rank_{r}"]["W_F4_24cell_1152"] for r in ranks]
    d_o4 = [inv[f"rank_{r}"]["O4_continuous"] for r in ranks]

    w = 0.25
    b1 = ax.bar(
        ranks - w,
        d_wb4,
        width=w,
        color="#dc2626",
        alpha=0.86,
        label=("Hiperkockás $W(B_4)$ (384 elem, $r=4$ LIV)" if lang == "hu" else "Hypercubic $W(B_4)$ (384 elem., $r=4$ LIV)"),
    )
    b2 = ax.bar(
        ranks,
        d_wf4,
        width=w,
        color="#2563eb",
        alpha=0.92,
        label=("24-cellás $W(F_4)$ (1152 elem)" if lang == "hu" else "24-cell $W(F_4)$ (1,152 elem.)"),
    )
    b3 = ax.bar(
        ranks + w,
        d_o4,
        width=w,
        color="#0f172a",
        alpha=0.84,
        label=("Folytonos $O(4)$ Lorentz/euklideszi" if lang == "hu" else "Continuous $O(4)$ Lorentz/Euclidean"),
    )
    for bars in (b1, b2, b3):
        for rect in bars:
            hv = int(round(rect.get_height()))
            if hv > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    hv + 0.45,
                    f"{hv}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    fontweight="bold",
                )

    ax.axvspan(0.55, 5.45, color="#dbeafe", alpha=0.28, zorder=0)
    ax.text(
        0.03,
        0.46,
        (
            rf"Egzakt $W(F_4) \equiv O(4)$ azonosság $r \leq 5$-ig:" + "\n"
            rf"Minden $d \leq 7$ LIV-operátor $= 0$ (minden hurokban)" + "\n"
            rf"LHAASO $18\;\mathrm{{TeV}}$ ($d=8$): $|\Delta v_g/c| = {fmt_sci(g1['jacob_piran_d8_dvg_over_c_LHAASO_18TeV'], 2, lang)}$"
            if lang == "hu"
            else rf"Exact $W(F_4) \equiv O(4)$ identity for $r \leq 5$:" + "\n"
            rf"All $d \leq 7$ LIV operators $\equiv 0$ (to all loop orders)" + "\n"
            rf"LHAASO $18\;\mathrm{{TeV}}$ ($d=8$): $|\Delta v_g/c| = {fmt_sci(g1['jacob_piran_d8_dvg_over_c_LHAASO_18TeV'], 2, lang)}$"
        ),
        transform=ax.transAxes,
        fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1"),
    )
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"$r={r}$" for r in ranks])
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_ylim(0, 32.5)
    ax.set_ylabel("Invariáns tenzorok száma" if lang == "hu" else "Invariant tensor dimension")
    ax.set_title(
        "A. Kapu S3.1: 24-cellás $W(F_4)$ tenzorgyűrű-izotrópia"
        if lang == "hu"
        else "A. Gate S3.1: 24-Cell $W(F_4)$ Tensor-Ring Isotropy"
    )
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    ax.legend(loc="upper left", fontsize=7.1, framealpha=0.94)

    # --- Panel B: Infinite-Lattice D_4 Spectral Dimension d_S(sigma) & Pin(W(F_4)) ---
    ax = axes[0, 1]
    sig_grid = np.array(g1["dS_sigma_grid"], dtype=float)
    ds_vals = np.array(g1["dS_curve_vals"], dtype=float)
    sig_fine = np.geomspace(2.5, 100.0, 200)
    ds_asymp = 4.0 + 0.5 / sig_fine

    ax.semilogx(
        sig_grid,
        ds_vals,
        "o-",
        color="#2563eb",
        lw=2.1,
        ms=5.0,
        label=("Egzakt végtelen $\\Lambda_{D_4}$ rács $d_S(\\sigma)$" if lang == "hu" else "Exact infinite $\\Lambda_{D_4}$ lattice $d_S(\\sigma)$"),
    )
    ax.semilogx(
        sig_fine,
        ds_asymp,
        "--",
        color="#dc2626",
        lw=1.5,
        label=("Analitikus IR-határ $4 + 1/(2\\sigma)$" if lang == "hu" else "Analytic IR asymptote $4 + 1/(2\\sigma)$"),
    )
    ax.axhline(
        4.0,
        color="#0f172a",
        ls=":",
        lw=1.3,
        label=("4D kontinuum határ ($d_S = 4$)" if lang == "hu" else "4D continuum limit ($d_S = 4$)"),
    )
    n_plaq = g1["triangular_plaquettes_per_site"]
    dt_span = g1["max_euclidean_time_slice_span"]
    ax.text(
        0.39,
        0.43,
        (
            rf"UV-csúcs $d_S(0{{,}}5) = {fmt_num(g1['spectral_dimension_sigma05_UV_peak'], 4, lang, True)}$; $d_S(100) = {fmt_num(g1['spectral_dimension_sigma100'], 4, lang, True)}$" + "\n"
            rf"$\mathrm{{Pin}}(W(F_4))$ Overlap hiba: ${fmt_sci(g1['pin4_overlap_covariance_max_err_1152'], 2, lang)}$" + "\n"
            rf"1-hurok BZ LIV ($d=4;\;6$): ${fmt_sci(g1['oneloop_BZ_d4_anisotropy'], 1, lang)}$" + "\n"
            rf"${n_plaq}$ háromszög-plakett/csúcs: $|\Delta n_0| \leq {dt_span}$ (OS-pozitív)"
            if lang == "hu"
            else rf"UV peak $d_S(0.5) = {g1['spectral_dimension_sigma05_UV_peak']:.4f}$; $d_S(100) = {g1['spectral_dimension_sigma100']:.4f}$" + "\n"
            rf"$\mathrm{{Pin}}(W(F_4))$ Overlap err: ${fmt_sci(g1['pin4_overlap_covariance_max_err_1152'], 2, lang)}$" + "\n"
            rf"1-loop BZ LIV ($d=4,6$): ${fmt_sci(g1['oneloop_BZ_d4_anisotropy'], 1, lang)}$" + "\n"
            rf"${n_plaq}$ triangle plaquettes/site: $|\Delta n_0| \leq {dt_span}$ (OS positive)"
        ),
        transform=ax.transAxes,
        fontsize=6.95,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1"),
    )
    ax.set_xlabel(r"Diffúziós sajátidő $\sigma / a^2$" if lang == "hu" else r"Diffusion proper time $\sigma / a^2$")
    ax.set_ylabel(r"Spektrális dimenzió $d_S(\sigma)$" if lang == "hu" else r"Spectral dimension $d_S(\sigma)$")
    ax.set_ylim(3.35, 5.75)
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_log_comma_formatter())
        ax.yaxis.set_major_formatter(make_comma_formatter(2))
    ax.set_title(
        "B. Spektrális dimenzió $d_S(\\sigma) \\to 4$ és OS-pozitivitás"
        if lang == "hu"
        else "B. Spectral Dimension $d_S(\\sigma) \\to 4$ & OS Positivity"
    )
    ax.grid(True, which="both", alpha=0.22, ls=":")
    ax.legend(loc="upper right", fontsize=7.2, framealpha=0.94)

    # --- Panel C: Gate S3.2 Local S_4 x S_4 Gauge Shield Curing the 7.40e13x Planck Stiffness Trap ---
    ax = axes[1, 0]
    cats = (
        [
            "Védetlen Planck\nmerevség (KIZÁRVA)",
            "Levezetett $c_2$\n($M_{\\mathrm{inf}}^2/M_{\\mathrm{Pl}}^2$)",
            "1-hurok SM\n$\\delta c_{\\mathrm{SM}}$ eltolás",
            "$S_4\\times S_4$ csupasz\ndirektorcsere",
        ]
        if lang == "hu"
        else [
            "Unprotected Planck\nstiffness (NO-GO)",
            "Derived $c_2$\n($M_{\\mathrm{inf}}^2/M_{\\mathrm{Pl}}^2$)",
            "1-loop SM\n$\\delta c_{\\mathrm{SM}}$ shift",
            "$S_4\\times S_4$ bare\ndirector exchange",
        ]
    )
    vals_c = [
        g2["planck_stiffness_trap_ratio_unprotected"],
        g2["derived_c2_Stueckelberg_Lorenz"],
        g2["derived_1loop_SM_lightspeed_shift"],
        1.0e-26,
    ]
    cols_c = ["#dc2626", "#2563eb", "#0ea5e9", "#16a34a"]
    y_pos = np.arange(len(cats))
    ax.barh(y_pos, [math.log10(v) for v in vals_c], color=cols_c, height=0.50, alpha=0.88)
    ax.axvline(0.0, color="#0f172a", lw=1.1, ls="--")
    xi_4 = g2["unprotected_healing_length_Mpc"] / 1.0e4
    ann_c = (
        [
            rf"${fmt_sci(g2['planck_stiffness_trap_ratio_unprotected'], 2, lang)}\times$ ($\xi = {fmt_num(xi_4, 2, lang, True)}\times 10^4\;\mathrm{{Mpc}}$)",
            rf"$c_2 = {fmt_sci(g2['derived_c2_Stueckelberg_Lorenz'], 2, lang)}$ ($\alpha_2^{{\mathrm{{PPN}}}}\equiv 0$)",
            rf"$\delta c_{{\mathrm{{SM}}}} = {fmt_sci(g2['derived_1loop_SM_lightspeed_shift'], 2, lang)}$",
            r"$\equiv 0$ (576 elemű $S_4\times S_4$ Reynolds-tétel)",
        ]
        if lang == "hu"
        else [
            rf"${fmt_sci(g2['planck_stiffness_trap_ratio_unprotected'], 2, lang)}\times$ ($\xi = {xi_4:.2f}\times 10^4\;\mathrm{{Mpc}}$)",
            rf"$c_2 = {fmt_sci(g2['derived_c2_Stueckelberg_Lorenz'], 2, lang)}$ ($\alpha_2^{{\mathrm{{PPN}}}}\equiv 0$)",
            rf"$\delta c_{{\mathrm{{SM}}}} = {fmt_sci(g2['derived_1loop_SM_lightspeed_shift'], 2, lang)}$",
            r"$\equiv 0$ (576-elem. $S_4\times S_4$ Reynolds theorem)",
        ]
    )
    for idx_y, (yv, lv) in enumerate(zip(y_pos, [math.log10(v) for v in vals_c])):
        if lv > 0:
            ax.text(-27.2, yv, ann_c[idx_y], va="center", ha="left", fontsize=7.0, color="#991b1b", fontweight="bold")
        else:
            ax.text(0.8, yv, ann_c[idx_y], va="center", ha="left", fontsize=7.0, color="#0f172a")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cats, fontsize=7.6)
    ax.set_ylim(-0.62, 3.62)
    ax.invert_yaxis()
    ax.set_xlim(-28.5, 31.0)
    ax.set_xlabel(r"$\log_{10}(\mathrm{dimenziótlan\;arány})$" if lang == "hu" else r"$\log_{10}(\mathrm{dimensionless\;scale\;ratio})$")
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_comma_formatter(0))
    ax.set_title(
        "C. Kapu S3.2: Lokális $S_4\\times S_4$ pajzs a Planck-befagyás ellen"
        if lang == "hu"
        else "C. Gate S3.2: Local $S_4\\times S_4$ Shield Against Planck Freezing"
    )
    ax.grid(True, axis="x", alpha=0.25, ls=":")

    # --- Panel D: Zero-Parameter Symbolic Derivation Summary from Single K_4 Generator u = sqrt(3)/24 ---
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title(
        "D. Zéró-paraméteres kontinuum-csatolási lánc ($u = \\sqrt{3}/24$)"
        if lang == "hu"
        else "D. Zero-Parameter Continuum Coupling Chain ($u = \\sqrt{3}/24$)"
    )

    u_str = fmt_num(g2["u_generator"], 8, lang, True)
    rows_d = (
        [
            (r"1. Alap $K_4$ szimplex-generátor", rf"$u = \frac{{\sqrt{{3}}}}{{24}} = {u_str}$", r"$r_{\mathrm{seed}} = 2\sqrt{3}\,u = \frac{1}{4}$"),
            (r"2. $T_{\mathrm{def}}$ spektrum ($\mathrm{Tr}\equiv 0$)", r"$(+5u,\; -3u,\; -u,\; -u)$", r"$[R_{SO(2)}, T_{\mathrm{def}}] \equiv 0$"),
            (r"3. Szingulett OCSR óra-súly", r"$\alpha_{\mathrm{OCSR}} = 5u = \frac{5\sqrt{3}}{24}$", rf"$= {fmt_num(g2['derived_alpha_OCSR'], 6, lang, True)}$"),
            (r"4. Nemminimális Helmholtz-pajzs", r"$\kappa_\theta = \xi_\Phi^2 = (256\pi^2)^2$", rf"$= {fmt_sci(g2['derived_kappa_theta'], 3, lang)}$"),
            (r"5. Stueckelberg–Lorenz merevség", r"$c_2 = \frac{c_{\mathrm{acc}}}{1 - 2c_{\mathrm{acc}}},\; c_S^2 \equiv 1$", rf"$= {fmt_sci(g2['derived_c2_Stueckelberg_Lorenz'], 3, lang)}$"),
            (r"6. Kondenzátum/barion arány", r"$R_{A/b} = \frac{2}{\alpha_{\mathrm{OCSR}}} - c_{\mathrm{mix}}$", rf"$= {fmt_num(g2['derived_R_Ab'], 6, lang, True)}$"),
        ]
        if lang == "hu"
        else [
            (r"1. Discrete $K_4$ generator", rf"$u = \frac{{\sqrt{{3}}}}{{24}} = {u_str}$", r"$r_{\mathrm{seed}} = 2\sqrt{3}\,u = \frac{1}{4}$"),
            (r"2. $T_{\mathrm{def}}$ spectrum ($\mathrm{Tr}\equiv 0$)", r"$(+5u,\; -3u,\; -u,\; -u)$", r"$[R_{SO(2)}, T_{\mathrm{def}}] \equiv 0$"),
            (r"3. Singlet OCSR clock weight", r"$\alpha_{\mathrm{OCSR}} = 5u = \frac{5\sqrt{3}}{24}$", rf"$= {g2['derived_alpha_OCSR']:.6f}$"),
            (r"4. Non-minimal Helmholtz shield", r"$\kappa_\theta = \xi_\Phi^2 = (256\pi^2)^2$", rf"$= {fmt_sci(g2['derived_kappa_theta'], 3, lang)}$"),
            (r"5. Stueckelberg–Lorenz stiffness", r"$c_2 = \frac{c_{\mathrm{acc}}}{1 - 2c_{\mathrm{acc}}},\; c_S^2 \equiv 1$", rf"$= {fmt_sci(g2['derived_c2_Stueckelberg_Lorenz'], 3, lang)}$"),
            (r"6. Condensate-to-baryon ratio", r"$R_{A/b} = \frac{2}{\alpha_{\mathrm{OCSR}}} - c_{\mathrm{mix}}$", rf"$= {g2['derived_R_Ab']:.6f}$"),
        ]
    )

    y_top = 0.90
    dy = 0.145
    for idx_r, (col1, col2, col3) in enumerate(rows_d):
        yc = y_top - idx_r * dy
        bg = "#eff6ff" if idx_r % 2 == 0 else "#f8fafc"
        rect = plt.Rectangle((0.01, yc - 0.055), 0.98, 0.118, transform=ax.transAxes, fc=bg, ec="#cbd5e1", lw=0.8)
        ax.add_patch(rect)
        ax.text(0.025, yc, col1, transform=ax.transAxes, va="center", ha="left", fontsize=7.3, fontweight="bold", color="#0f172a")
        ax.text(0.505, yc, col2, transform=ax.transAxes, va="center", ha="left", fontsize=7.5, color="#1e3a8a")
        ax.text(0.975, yc, col3, transform=ax.transAxes, va="center", ha="right", fontsize=7.5, color="#047857")

    out_path = FIG_OUT / ("fig5_24cell_isotropy_and_s4_planck_shield_hu.pdf" if lang == "hu" else "fig5_24cell_isotropy_and_s4_planck_shield.pdf")
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return out_path


def generate_fig6(res: Dict[str, Any], lang: str) -> Path:
    g3 = res["gate_S3_3_frullani_rg_and_simplicial_gamma_convergence"]
    g4 = res["gate_S3_4_void_clock_homogenization_and_epistemic_ledger"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 9.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.935, bottom=0.085, wspace=0.40, hspace=0.38)

    # --- Panel A: Gate S3.3 60-Decade Wetterich-Frullani Threshold Integral & Ellipticity ---
    ax = axes[0, 0]
    s_grid = np.array(g3["frullani_s_grid"], dtype=float)
    f_od = np.array(g3["frullani_FOD_vals"], dtype=float)
    lam_p = np.array(g3["frullani_lambda_par_vals"], dtype=float)

    ax.semilogx(
        s_grid,
        f_od,
        color="#2563eb",
        lw=2.2,
        label=("Frullani $F_{\\mathrm{OD}}(s)$ (3 regulátor)" if lang == "hu" else "Frullani $F_{\\mathrm{OD}}(s)$ (3 regulators)"),
    )
    ax.semilogx(
        s_grid,
        lam_p,
        color="#16a34a",
        ls="--",
        lw=1.9,
        label=(
            rf"Elliptikus sajátérték $\lambda_\parallel(s) \geq {fmt_num(g3['min_elliptic_eigenvalue_lambda_par'], 3, lang, True)}$"
            if lang == "hu"
            else rf"Elliptic eigenvalue $\lambda_\parallel(s) \geq {g3['min_elliptic_eigenvalue_lambda_par']:.3f}$"
        ),
    )
    ax.axhline(0.0, color="#dc2626", ls=":", lw=1.0)
    ax.text(
        0.44,
        0.47,
        (
            rf"SymPy 3-regulátor maradék $\equiv 0$" + "\n"
            rf"60 évtizedes RG hiba: ${fmt_sci(g3['max_60_decade_quadrature_err'], 2, lang)}$" + "\n"
            rf"Szigorúan konvex: $\lambda_\parallel(s) > 0$ mindenütt"
            if lang == "hu"
            else rf"SymPy 3-regulator residual $\equiv 0$" + "\n"
            rf"60-decade RG error: ${fmt_sci(g3['max_60_decade_quadrature_err'], 2, lang)}$" + "\n"
            rf"Strictly convex: $\lambda_\parallel(s) > 0$ everywhere"
        ),
        transform=ax.transAxes,
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1"),
    )
    ax.set_xlabel(r"$s = |\nabla\Phi_N| / a_0^{\mathrm{KH}}(z)$")
    ax.set_ylabel(r"$F_{\mathrm{OD}}(s)$ / $\lambda_\parallel(s)$")
    ax.set_ylim(-0.3, 6.2)
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_log_comma_formatter())
        ax.yaxis.set_major_formatter(make_comma_formatter(1))
    ax.set_title(
        "A. Kapu S3.3: 60 évtizedes Frullani-RG és ellipticitás"
        if lang == "hu"
        else "A. Gate S3.3: 60-Decade Frullani RG & Strict Ellipticity"
    )
    ax.grid(True, which="both", alpha=0.22, ls=":")
    ax.legend(loc="upper right", fontsize=7.3, framealpha=0.94)

    # --- Panel B: 3D Kuhn-Simplicial O(a^2) Gamma-Convergence (up to 196,608 Tetrahedra) ---
    ax = axes[0, 1]
    sc = g3["simplicial_gamma_convergence"]
    h_vals = np.array([row["cell_edge_box_units"] for row in sc], dtype=float)
    e_vals = np.array([row["rel_L2_err_vs_continuum"] for row in sc], dtype=float)
    n_tets = [row["n_tetrahedra"] for row in sc]

    h_ref_line = np.geomspace(h_vals[-1] * 0.85, h_vals[0] * 1.15, 50)
    e_ref_line = e_vals[-1] * ((h_ref_line / h_vals[-1]) ** 2)

    ax.loglog(
        h_vals,
        e_vals,
        "o-",
        color="#2563eb",
        lw=2.2,
        ms=6.2,
        label=("3D Kuhn-szimplex $P_1$ $L^2$-hiba" if lang == "hu" else "3D Kuhn-simplicial $P_1$ $L^2$ error"),
    )
    ax.loglog(
        h_ref_line,
        e_ref_line,
        "--",
        color="#0f172a",
        lw=1.5,
        label=("Ideális másodrendű $\\mathcal{O}(a^2)$ meredekség" if lang == "hu" else "Ideal second-order $\\mathcal{O}(a^2)$ slope"),
    )
    for idx_pt, (hv, ev, nt) in enumerate(zip(h_vals, e_vals, n_tets)):
        nt_str = f"{nt:,}".replace(",", "\\,") if lang == "hu" else f"{nt:,}"
        lbl_t = rf"${nt_str}$ tetraéder" if lang == "hu" else rf"${nt_str}$ tets"
        if idx_pt == 2:
            dx_pt, dy_pt, ha_pt, va_pt = (9, -4, "left", "top")
        elif idx_pt == 1:
            dx_pt, dy_pt, ha_pt, va_pt = (-8, 12, "center", "bottom")
        else:
            dx_pt, dy_pt, ha_pt, va_pt = (-8, 10, "right", "bottom")
        ax.annotate(
            lbl_t,
            xy=(hv, ev),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            ha=ha_pt,
            va=va_pt,
            fontsize=7.1,
            color="#1e3a8a",
            fontweight="bold",
        )

    ax.text(
        0.04,
        0.60,
        (
            rf"Diadikus arányok: $({fmt_num(g3['dyadic_ratio_8_to_16'], 2, lang, True)};\; {fmt_num(g3['dyadic_ratio_16_to_32'], 2, lang, True)})$" + "\n"
            rf"Befagyasztott direktor (KIZÁRVA): $|F_{{\mathrm{{net}}}}| = {fmt_sci(g3['fnet_frozen_director_NOGO'], 2, lang)}$" + "\n"
            rf"Teljes Noether-hatás: $|F_{{\mathrm{{net}}}}| = {fmt_sci(g3['fnet_full_noether_action'], 2, lang)}$"
            if lang == "hu"
            else rf"Dyadic ratios: $({g3['dyadic_ratio_8_to_16']:.2f},\; {g3['dyadic_ratio_16_to_32']:.2f})$" + "\n"
            rf"Frozen director (NO-GO): $|F_{{\mathrm{{net}}}}| = {fmt_sci(g3['fnet_frozen_director_NOGO'], 2, lang)}$" + "\n"
            rf"Full Noether action: $|F_{{\mathrm{{net}}}}| = {fmt_sci(g3['fnet_full_noether_action'], 2, lang)}$"
        ),
        transform=ax.transAxes,
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.26", fc="#f8fafc", ec="#cbd5e1"),
    )
    ax.set_xlabel(r"Szimplex rácslépték $a$ [rács-egység]" if lang == "hu" else r"Simplicial mesh spacing $a$ [box units]")
    ax.set_ylabel(r"$\|\Phi_{\mathrm{tot}}^{P_1} - \Phi_{\mathrm{tot}}^{\mathrm{cont}}\|_{L^2} / \|\Phi_{\mathrm{tot}}^{\mathrm{cont}}\|_{L^2}$")
    ax.set_xlim(0.92, 6.40)
    ax.set_ylim(1.20e-3, 0.35)
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_log_comma_formatter())
        ax.yaxis.set_major_formatter(make_log_comma_formatter())
    ax.set_title(
        "B. 3D szimplex $\\Gamma$-konvergencia ($196\\,608$ tetraéderig)"
        if lang == "hu"
        else "B. 3D Simplicial $\\Gamma$-Convergence (Up to $196{,}608$ Tets)"
    )
    ax.grid(True, which="both", alpha=0.22, ls=":")
    ax.legend(loc="lower right", fontsize=7.3, framealpha=0.94)

    # --- Panel C: Gate S3.4 64^3 Cosmic-Web Void-Clock Homogenization ---
    ax = axes[1, 0]
    zg = np.array(g4["homogenization_z_grid"], dtype=float)
    w_wilt = np.array(g4["normalized_void_weight_wiltshire"], dtype=float)
    w_shrp = np.array(g4["normalized_void_weight_sharp_step"], dtype=float)
    w_d10 = np.array(g4["normalized_void_weight_smooth_d010"], dtype=float)

    band_frac = g4["step1_wilks_1sigma_band_pct"] / 100.0
    ax.fill_between(
        zg,
        w_wilt * (1.0 - band_frac),
        np.minimum(1.12, w_wilt * (1.0 + band_frac)),
        color="#dbeafe",
        alpha=0.55,
        label=(
            rf"1. lépés Wilks $1\sigma$ sáv ($\pm {fmt_num(g4['step1_wilks_1sigma_band_pct'], 1, lang, True)}\%$)"
            if lang == "hu"
            else rf"Step-1 Wilks $1\sigma$ band ($\pm {g4['step1_wilks_1sigma_band_pct']:.1f}\%$)"
        ),
    )
    ax.plot(
        zg,
        w_wilt,
        "o-",
        color="#2563eb",
        lw=2.1,
        ms=4.8,
        label=("Dinamikus Wiltshire-követő $f_v(z)/f_{v,0}$" if lang == "hu" else "Dynamical Wiltshire tracker $f_v(z)/f_{v,0}$"),
    )
    ax.plot(
        zg,
        w_shrp,
        "s--",
        color="#0f172a",
        lw=1.7,
        ms=4.4,
        label=(
            rf"$64^3$ lognormál éles küszöb ($\eta={fmt_num(g4['underdense_void_threshold_rho_over_bar'], 3, lang, True)}$)"
            if lang == "hu"
            else rf"$64^3$ log-normal sharp step ($\eta={g4['underdense_void_threshold_rho_over_bar']:.3f}$)"
        ),
    )
    ax.plot(
        zg,
        w_d10,
        "d:",
        color="#dc2626",
        lw=1.7,
        ms=4.4,
        label=("Sima filamentfal ($\\Delta\\eta = 0{,}10$)" if lang == "hu" else "Smooth filament wall ($\\Delta\\eta = 0.10$)"),
    )
    ax.text(
        0.03,
        0.08,
        (
            rf"512 ergodikus fénysugár hiba: ${fmt_num(100.0 * g4['ergodic_ray_vs_volume_rel_err'], 2, lang, True)}\%$" + "\n"
            rf"Sima fal eltolás ($z=0{{,}}5$): $+{fmt_num(g4['smooth_d010_filament_shift_pct_at_z05'], 2, lang, True)}\%$" + "\n"
            rf"Statikus vs. Wiltshire ($z=0{{,}}5$): $+{fmt_num(g4['static_lognormal_vs_wiltshire_tracker_pct_at_z05'], 2, lang, True)}\%$"
            if lang == "hu"
            else rf"512 ergodic null-ray err: ${100.0 * g4['ergodic_ray_vs_volume_rel_err']:.2f}\%$" + "\n"
            rf"Smooth wall shift ($z=0.5$): $+{g4['smooth_d010_filament_shift_pct_at_z05']:.2f}\%$" + "\n"
            rf"Static LN vs. Wiltshire ($z=0.5$): $+{g4['static_lognormal_vs_wiltshire_tracker_pct_at_z05']:.2f}\%$"
        ),
        transform=ax.transAxes,
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1"),
    )
    ax.set_xlabel("Vöröseltolódás $z$" if lang == "hu" else "Redshift $z$")
    ax.set_ylabel(r"Normált üreg-súly $W_{\mathrm{void}}(z)$" if lang == "hu" else r"Normalized void weight $W_{\mathrm{void}}(z)$")
    ax.set_ylim(0.22, 1.16)
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_comma_formatter(1))
        ax.yaxis.set_major_formatter(make_comma_formatter(2))
    ax.set_title(
        "C. Kapu S3.4: $64^3$ kozmikus háló üreg-óra homogenizáció"
        if lang == "hu"
        else "C. Gate S3.4: $64^3$ Cosmic-Web Void-Clock Homogenization"
    )
    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(loc="upper right", fontsize=7.0, framealpha=0.94)

    # --- Panel D: Complete Paper 29 Three-Pillar Epistemic Ledger Overview ---
    ax = axes[1, 1]
    pillars = (
        ["1. pillér\n($\\theta_*$–BAO–SN)", "2. pillér\n(Bullet + 64 halmaz)", "3. pillér\n(24-cella $\\to$ EFT)"]
        if lang == "hu"
        else ["Pillar 1\n($\\theta_*$–BAO–SN)", "Pillar 2\n(Bullet + 64 clust.)", "Pillar 3\n(24-cell $\\to$ EFT)"]
    )
    # Gate/claim counts per epistemic category loaded dynamically from JSON:
    cnt = res["three_pillar_epistemic_counts"]
    p_keys = ("pillar_1", "pillar_2", "pillar_3")
    n_proven = np.array([cnt[pk]["proven"] for pk in p_keys], dtype=int)
    n_nogo = np.array([cnt[pk]["nogo"] for pk in p_keys], dtype=int)
    n_cond = np.array([cnt[pk]["conditional"] for pk in p_keys], dtype=int)

    xp = np.arange(len(pillars))
    bw = 0.24
    p1 = ax.bar(xp - bw, n_proven, width=bw, color="#16a34a", label=("IGAZOLT (tétel / konkordancia)" if lang == "hu" else "PROVEN (theorem / concordance)"))
    p2 = ax.bar(xp, n_nogo, width=bw, color="#dc2626", label=("KIZÁRVA (alternatíva megcáfolva)" if lang == "hu" else "PROVEN NO-GO (falsified alt.)"))
    p3 = ax.bar(xp + bw, n_cond, width=bw, color="#d97706", label=("FELTÉTELES / UV-axióma határ" if lang == "hu" else "CONDITIONAL / UV-axiom scope"))
    for bars in (p1, p2, p3):
        for rect in bars:
            hv = int(round(rect.get_height()))
            ax.text(rect.get_x() + rect.get_width() / 2.0, hv + 0.10, f"{hv}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    ax.text(
        0.025,
        0.65,
        (
            "Irredukálható UV-axiómák:\n"
            "• AXIOM-OCC: $|\\nu|=1$ $K_4 \\to 24$-cella ($r_{\\mathrm{seed}}=1/4$)\n"
            "• AXIOM-KMS: Kodama–Hayward $a_0^{\\mathrm{KH}}(z)$\n"
            "• AXIOM-FRG: LPA' Frullani $s\\sim 1$ átmenet"
            if lang == "hu"
            else "Irreducible UV axioms:\n"
            "• AXIOM-OCC: $|\\nu|=1$ $K_4 \\to 24$-cell ($r_{\\mathrm{seed}}=1/4$)\n"
            "• AXIOM-KMS: Kodama–Hayward $a_0^{\\mathrm{KH}}(z)$\n"
            "• AXIOM-FRG: LPA' Frullani $s\\sim 1$ crossover"
        ),
        transform=ax.transAxes,
        fontsize=6.85,
        bbox=dict(boxstyle="round,pad=0.23", fc="#fffbeb", ec="#fcd34d"),
    )
    ax.set_xticks(xp)
    ax.set_xticklabels(pillars, fontsize=8.1)
    ax.set_ylim(0, 7.6)
    ax.set_ylabel("Ellenőrzött tételek / kapuk száma" if lang == "hu" else "Verified theorems / sub-gates")
    ax.set_title(
        "D. A Paper 29 teljes 3-pilléres episztemikus mérlege"
        if lang == "hu"
        else "D. Complete 3-Pillar Epistemic Ledger of Paper 29"
    )
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    ax.legend(loc="upper right", fontsize=7.0, framealpha=0.94)

    out_path = FIG_OUT / (
        "fig6_frullani_rg_simplicial_gamma_and_void_homogenization_hu.pdf"
        if lang == "hu"
        else "fig6_frullani_rg_simplicial_gamma_and_void_homogenization.pdf"
    )
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return out_path


def main() -> None:
    res = json.loads(JSON_IN.read_text(encoding="utf-8"))
    for lg in ("en", "hu"):
        p5 = generate_fig5(res, lg)
        p6 = generate_fig6(res, lg)
        print(f"[{lg.upper()}] Generated {p5.name} & {p6.name}")


if __name__ == "__main__":
    main()
