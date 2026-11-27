#!/usr/bin/env python3
"""
JSON-Driven Vector PDF Figure Generator (EN + HU) for Tau Core Paper 29 — Step 1:
  - figures/fig1_ltb_weinberg_nogo_and_disformal_sachs[_hu].pdf
  - figures/fig2_zero_param_theta_bao_sn_boltzmann_3000[_hu].pdf

Strict Referee Formatting Rules:
  - Must pass `python3 -W error` cleanly with exit code 0.
  - 100% Hungarian decimal commas on all `_hu.pdf` tick labels and annotations.
  - Uses `\\leq` / `\\geq` (never `\\le` / `\\ge`) for Matplotlib mathtext compatibility.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy import integrate, interpolate

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
JSON_IN = ROOT_DIR / "paper29_step1_verification_results.json"


def fmt_num(val: float, dec: int, lang: str) -> str:
    s = f"{val:.{dec}f}"
    return s.replace(".", "{,}") if lang == "hu" else s


def fmt_plain(val: float, dec: int, lang: str) -> str:
    s = f"{val:.{dec}f}"
    return s.replace(".", ",") if lang == "hu" else s


def fmt_sci(val: float, dec: int, lang: str) -> str:
    exp = int(math.floor(math.log10(abs(val)))) if val != 0.0 else 0
    mant = val / (10.0**exp)
    m_s = f"{mant:.{dec}f}".replace(".", "{,}") if lang == "hu" else f"{mant:.{dec}f}"
    return f"{m_s} \\times 10^{{{exp}}}"


def make_comma_formatter(dec: int = 2):
    def _fmt(x: float, _pos: int) -> str:
        if abs(x - round(x)) < 1.0e-9 and dec == 0:
            return f"{int(round(x))}"
        if abs(x) >= 100.0 and dec <= 1:
            return f"{int(round(x))}"
        return f"{x:.{dec}f}".replace(".", ",")
    return FuncFormatter(_fmt)


def build_fig1(res: Dict, lang: str) -> None:
    g1 = res["gate_S1_1_ltb_swiss_cheese_weinberg_nogo"]
    g2 = res["gate_S1_2_disformal_optical_metric_proof"]
    g3 = res["gate_S1_3_untruncated_boltzmann_3000_bao_sn_closure"]
    g4 = res["gate_S1_4_local_void_shoes_vs_jwst_ceiling"]
    lc = res["locked_constants"]

    plt.rcParams.update({"font.size": 9.2, "axes.titlesize": 9.8, "axes.labelsize": 9.2})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.3))
    fig.subplots_adjust(hspace=0.33, wspace=0.38, left=0.072, right=0.985, top=0.93, bottom=0.08)

    # Reconstruct exact LTB Eulerian radial profile at z=0 for Panel A:
    Om_m = g1["Omega_m0"]
    Om_L = 1.0 - Om_m
    a_ini = 1.0e-3
    di = -3.25e-3
    u_sh = np.linspace(1.0e-4, 1.0, 400)
    W_u = (1.0 - u_sh**2) ** 3
    dW = -6.0 * u_sh * ((1.0 - u_sh**2) ** 2)
    db = di * W_u
    dl = di * (W_u + (u_sh / 3.0) * dW)
    E_ini = math.sqrt(Om_m / (a_ini**3) + Om_L)
    dy_ini = a_ini * E_ini * (1.0 - db / 3.0)
    kt = Om_m * (1.0 + db) / a_ini + Om_L * (a_ini**2) - (dy_ini**2)
    kt[-1] = 0.0
    sol = integrate.solve_ivp(
        lambda l, y: np.sqrt(np.maximum(Om_m * (1.0 + db) / y - kt + Om_L * y * y, 1.0e-30)) / math.sqrt(Om_m / (math.exp(l) ** 3) + Om_L),
        (math.log(a_ini), 0.0),
        np.full(400, a_ini),
        t_eval=[0.0],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    yv = sol.y[:, 0]
    Ap = yv + u_sh * np.gradient(yv, u_sh)
    rr = (1.0 + dl) / ((yv**2) * Ap) - 1.0
    re = (1.0 + db) / (yv**3) - 1.0
    ue = yv * u_sh

    # Panel A
    ax = axes[0, 0]
    ax.axhline(0.0, color="#64748b", ls="--", lw=1.0)
    ax.plot(
        ue * g1["R_cell_Mpc"],
        rr,
        color="#2563eb",
        lw=2.1,
        label=(
            r"Lokális $\delta_{\mathrm{LTB}}(r_{\mathrm{eul}}, z=0)$"
            if lang == "hu"
            else r"Local $\delta_{\mathrm{LTB}}(r_{\mathrm{eul}}, z=0)$"
        ),
    )
    ax.plot(
        ue * g1["R_cell_Mpc"],
        re - rr,
        color="#d97706",
        lw=1.7,
        ls="-.",
        label=(
            r"Weyl árapály-forrás $(\bar{\rho}_{\mathrm{enc}}-\rho)/\bar{\rho}$"
            if lang == "hu"
            else r"Weyl tidal source $(\bar{\rho}_{\mathrm{enc}}-\rho)/\bar{\rho}$"
        ),
    )
    ax.set_xlim(0.0, 50.0)
    ax.set_ylim(-0.92, 0.78)
    ax.set_xlabel("Euler-féle társamozgó sugár $r_{\\mathrm{eul}}$ [Mpc]" if lang == "hu" else "Eulerian comoving radius $r_{\\mathrm{eul}}$ [Mpc]")
    ax.set_ylabel("Sűrűségkontraszt" if lang == "hu" else "Density contrast")
    ax.set_title(
        "A. Egzakt 4D $C^2$ LTB Swiss-Cheese üreg–fal profil ($z=0$)"
        if lang == "hu"
        else "A. Exact 4D $C^2$ LTB Swiss-Cheese Void-Wall Profile ($z=0$)"
    )
    txt_a = (
        f"$\\delta_0 = {fmt_num(g1['delta_void_center_z0'], 3, lang)}$, fal: $+{fmt_num(g1['delta_wall_peak_z0'], 3, lang)}$\n"
        f"Tömegkomp. hiba: ${fmt_sci(g1['mass_compensation_rel_err_z0'], 2, lang)}$"
        if lang == "hu"
        else f"$\\delta_0 = {g1['delta_void_center_z0']:.3f}$, wall: $+{g1['delta_wall_peak_z0']:.3f}$\n"
        f"Mass-comp. error: ${fmt_sci(g1['mass_compensation_rel_err_z0'], 2, lang)}$"
    )
    ax.text(0.04, 0.83, txt_a, transform=ax.transAxes, fontsize=8.2, bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1"))
    ax.legend(loc="lower right", fontsize=8.0, frameon=True)
    if lang == "hu":
        ax.yaxis.set_major_formatter(make_comma_formatter(1))
        ax.xaxis.set_major_formatter(make_comma_formatter(0))

    # Panel B: Weinberg Flux Conservation Bar Comparison (symlog scale showing extreme rays vs area mean = +0.0043%)
    ax = axes[0, 1]
    vals_b = [
        g1["D_A_void_center_rel_shift_pct"],
        g1["D_A_wall_shell_rel_shift_pct"],
        g1["D_A_ensemble_area_mean_rel_shift_pct"],
        -g1["required_theta_star_shift_pct"],
    ]
    labels_b = (
        ["Üreg-alagút\nsugár (extrém)", "Fal-alagút\nsugár (extrém)", "Égbolt-átlag\n(Weinberg)", "Szükséges\nDESI eltolás"]
        if lang == "hu"
        else ["Void-tunnel\nray (extreme)", "Wall-tunnel\nray (extreme)", "Area mean\n(Weinberg)", "Required\nDESI shift"]
    )
    cols_b = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444"]
    bars = ax.bar(np.arange(4), vals_b, color=cols_b, width=0.55, edgecolor="#1e293b", lw=0.8)
    ax.axhline(0.0, color="#334155", lw=1.0)
    ax.set_yscale("symlog", linthresh=0.10, linscale=1.2)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels_b, fontsize=8.0)
    ax.set_ylim(-55.0, 260.0)
    yticks_b = [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0]
    ax.set_yticks(yticks_b)
    ax.set_yticklabels([fmt_plain(y, 1 if abs(y) < 1.0 and y != 0.0 else 0, lang) for y in yticks_b], fontsize=8.0)
    ax.set_ylabel(r"$100 \times \Delta D_A(z_*) / D_A^{\mathrm{FLRW}}(z_*)$ [%]")
    ax.set_title(
        "B. Kapu S1.1: 4D LTB Weinberg-tétel NO-GO bizonyítás"
        if lang == "hu"
        else "B. Gate S1.1: 4D LTB Weinberg Flux-Conservation NO-GO"
    )
    y_lbl_pos = [115.0, -24.0, 0.025, -0.62]
    decs_b = [2, 2, 4, 3]
    for b_rect, v_b, yp_b, dc_b in zip(bars, vals_b, y_lbl_pos, decs_b):
        lbl_v = f"${v_b:+.{dc_b}f}\\%$".replace(".", "{,}") if lang == "hu" else f"${v_b:+.{dc_b}f}\\%$"
        ax.text(b_rect.get_x() + b_rect.get_width() / 2.0, yp_b, lbl_v, ha="center", va="center", fontsize=8.0, fontweight="bold")

    # Panel C: Disformal Clock-Lapse Metric N_clock(z), w_DE,eff(z), and Strict Null Convergence dE_opt/dz > 0
    ax = axes[1, 0]
    h_t = g3["void_clock_zero_param_branch_A_pure_optical"]["H0"] / 100.0
    Om_m0 = lc["omega_m_phys"] / (h_t * h_t)
    Om_r0 = 4.1838e-5 / (h_t * h_t)
    Om_L0 = 1.0 - Om_m0 - Om_r0
    roots = np.roots([8.0, 39.0, -12.0, -8.0])
    f_acc = float(np.real([r for r in roots if np.isreal(r) and 0 < np.real(r) < 1][0]))
    z_acc = ((2.0 * (1.0 - Om_m0)) / Om_m0) ** (1.0 / 3.0) - 1.0
    from scipy.optimize import brentq
    fv0 = float(brentq(lambda f: ((2 + f_acc) / (2 + f)) * ((f / f_acc) ** (2 / 3)) * ((1 - f_acc) / (1 - f)) - (1 + z_acc), 0.59, 0.92))
    fv_tab = fv0 * np.exp(-np.linspace(0.0, 10.0, 500))
    x0 = (2.0 + fv0) / 3.0
    b_dim = 2.0 * (1.0 - fv0) * (2.0 + fv0) / (9.0 * fv0)
    h_ratio = (4.0 * fv0 * fv0 + fv0 + 4.0) / (2.0 * (2.0 + fv0))
    zp1 = ((2.0 + fv_tab) / (2.0 + fv0)) * ((fv0 / fv_tab) ** (2.0 / 3.0)) * ((1.0 - fv_tab) / (1.0 - fv0))
    x = (1.0 - fv0) * (2.0 + fv0) * fv_tab / (3.0 * fv0 * (1.0 - fv_tab))
    x[0] = x0
    b13, x13 = b_dim ** (1.0 / 3.0), x ** (1.0 / 3.0)
    F_v = 2.0 * x13 + (b13 / 6.0) * np.log(((x13 + b13) ** 2) / (x13 * x13 - b13 * x13 + b13 * b13)) + (b13 / math.sqrt(3.0)) * np.arctan((2.0 * x13 - b13) / (math.sqrt(3.0) * b13))
    Hbar0_dA = (x ** (2.0 / 3.0)) * (float(F_v[0]) - F_v)
    dln1pz_dx = 1.0 / (x + b_dim) - 2.0 / (3.0 * x) - 8.0 / (3.0 * (2.0 * x + 3.0 * b_dim))
    dx_dz = 1.0 / (zp1 * dln1pz_dx)
    d_Hbar0_dA_dz = ((2.0 / (3.0 * x)) * Hbar0_dA - 2.0 / (2.0 + fv_tab)) * dx_dz
    dD_dz = h_ratio * (Hbar0_dA + zp1 * d_Hbar0_dA_dz)
    dD_dz[0] = 1.0
    u_tab = np.log(zp1)
    u_tab[0] = 0.0
    sp_dD = interpolate.CubicSpline(u_tab, dD_dz)
    sp_fv = interpolate.CubicSpline(u_tab, fv_tab)

    z_p = np.linspace(0.0, 2.5, 500)
    u_p = np.log1p(z_p)
    E_h = np.sqrt(Om_r0 * ((1.0 + z_p) ** 4) + Om_m0 * ((1.0 + z_p) ** 3) + Om_L0)
    w_v = lc["alpha_OCSR"] * (sp_fv(u_p) / fv0)
    N_cl = (1.0 - w_v) + w_v * E_h * sp_dD(u_p)
    E_op = E_h / N_cl
    dE_du = np.gradient(E_op, u_p)
    w_de = (-1.0 + (2.0 / 3.0) * dE_du / E_op - (Om_r0 * ((1.0 + z_p) ** 4)) / (3.0 * (E_op**2))) / (
        1.0 - (Om_r0 * ((1.0 + z_p) ** 4) + Om_m0 * ((1.0 + z_p) ** 3)) / (E_op**2)
    )
    dE_dz = dE_du / (1.0 + z_p)

    ax.axhline(-1.0, color="#64748b", ls=":", lw=1.1)
    ax.plot(z_p, w_de, color="#2563eb", lw=2.1, label=r"Látszólagos $w_{\mathrm{DE,eff}}(z)$" if lang == "hu" else r"Apparent $w_{\mathrm{DE,eff}}(z)$")
    ax.plot(z_p, dE_dz - 1.5, color="#059669", lw=1.8, ls="--", label=r"$\mathrm{d}E_{\mathrm{opt}}/\mathrm{d}z - 1{,}5$ (NCC $>0$)" if lang == "hu" else r"$\mathrm{d}E_{\mathrm{opt}}/\mathrm{d}z - 1.5$ (NCC $>0$)")
    ax.axvline(g2["apparent_phantom_crossing_z"], color="#dc2626", ls="-.", lw=1.0)
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(-1.32, -0.55)
    ax.set_xlabel("Vöröseltolódás $z$" if lang == "hu" else "Redshift $z$")
    ax.set_ylabel("Hatásos állapotegyenlet" if lang == "hu" else "Effective EoS / Convergence")
    ax.set_title(
        r"C. Kapu S1.2: 4D $\tilde{g}_{\mu\nu}^{\mathrm{opt}}$ Sachs-azonosság és NCC ($R_{\mu\nu}k^\mu k^\nu > 0$)"
        if lang == "hu"
        else r"C. Gate S1.2: 4D $\tilde{g}_{\mu\nu}^{\mathrm{opt}}$ Exact Sachs Identity & Strict NCC"
    )
    txt_c = (
        f"SymPy 4D Sachs maradék $= 0$\nDOP853 hiba: ${fmt_sci(g2['all_z_forward_sachs_rel_error'], 2, lang)}$\n"
        f"Fantom-átlépés: $z_\\times = {fmt_num(g2['apparent_phantom_crossing_z'], 2, lang)}$"
        if lang == "hu"
        else f"SymPy 4D Sachs residual $= 0$\nDOP853 error: ${fmt_sci(g2['all_z_forward_sachs_rel_error'], 2, lang)}$\n"
        f"Phantom crossing: $z_\\times = {g2['apparent_phantom_crossing_z']:.2f}$"
    )
    ax.text(0.42, 0.12, txt_c, transform=ax.transAxes, fontsize=8.1, bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1"))
    ax.legend(loc="upper right", fontsize=8.0)
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_comma_formatter(1))
        ax.yaxis.set_major_formatter(make_comma_formatter(2))

    # Panel D: Local H0 Ceiling Comparison (Gate S1.4)
    ax = axes[1, 1]
    y_pos_d = np.arange(5)
    h0_vals = [
        g4["H0_asymptotic_void_clock_bg"],
        g4["classix_120Mpc_2sigma_OG_ceiling"]["H0_loc_total"],
        g4["classix_120Mpc_2sigma_ruler_ceiling"]["H0_loc_total"],
        69.85,
        73.04,
    ]
    h0_errs = [0.0, 0.0, 0.0, 2.33, 1.04]
    lbls_d = (
        [
            "Tau Core háttér\n($\\theta_*$-rögzített)",
            "Lokális üreg $2\\sigma$ plafon\n(csak $\\mathcal{O}_G$ kiáramlás)",
            "Lokális üreg $2\\sigma$ plafon\n($\\mathcal{O}_G + \\mathcal{O}_T$ óra)",
            "CCHP JWST TRGB\n(Freedman+ 2025)",
            "SH0ES Cefeida\n(Riess+ 2022)",
        ]
        if lang == "hu"
        else [
            "Tau Core background\n($\\theta_*$-locked)",
            "Local void $2\\sigma$ ceiling\n($\\mathcal{O}_G$ outflow only)",
            "Local void $2\\sigma$ ceiling\n($\\mathcal{O}_G + \\mathcal{O}_T$ ruler)",
            "CCHP JWST TRGB\n(Freedman+ 2025)",
            "SH0ES Cepheid\n(Riess+ 2022)",
        ]
    )
    cols_d = ["#475569", "#2563eb", "#059669", "#d97706", "#dc2626"]
    for idx_d in range(5):
        if h0_errs[idx_d] > 0:
            ax.errorbar(h0_vals[idx_d], idx_d, xerr=h0_errs[idx_d], fmt="o", color=cols_d[idx_d], capsize=4, lw=1.8, ms=6)
        else:
            ax.plot(h0_vals[idx_d], idx_d, marker="D", color=cols_d[idx_d], ms=7)
        ax.text(
            h0_vals[idx_d] + (0.25 if idx_d < 3 else 0.0),
            idx_d + (0.22 if idx_d >= 3 else 0.0),
            fmt_plain(h0_vals[idx_d], 2, lang),
            fontsize=8.2,
            color=cols_d[idx_d],
            fontweight="bold",
        )
    ax.axvline(g4["classix_120Mpc_2sigma_ruler_ceiling"]["H0_loc_total"], color="#059669", ls="--", lw=1.1)
    ax.set_yticks(y_pos_d)
    ax.set_yticklabels(lbls_d, fontsize=8.0)
    ax.set_xlim(65.8, 75.2)
    ax.set_ylim(-0.6, 4.7)
    ax.set_xlabel(r"$H_0$ [$\mathrm{km\,s^{-1}\,Mpc^{-1}}$]")
    ax.set_title(
        "D. Kapu S1.4: Pantheon+SH0ES $1341\\times 1341$ lokális $H_0$ plafon"
        if lang == "hu"
        else "D. Gate S1.4: Pantheon+SH0ES $1341\\times 1341$ Local $H_0$ Ceiling"
    )
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_comma_formatter(0))

    out_name = "fig1_ltb_weinberg_nogo_and_disformal_sachs_hu.pdf" if lang == "hu" else "fig1_ltb_weinberg_nogo_and_disformal_sachs.pdf"
    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)


def build_fig2(res: Dict, lang: str) -> None:
    g3 = res["gate_S1_3_untruncated_boltzmann_3000_bao_sn_closure"]
    cv = g3["plot_curves"]
    vcA = g3["void_clock_zero_param_branch_A_pure_optical"]
    vcB = g3["void_clock_zero_param_branch_B_unified_drag"]
    hom_th = g3["homogeneous_theta_star_locked_baseline"]
    hom_68 = g3["homogeneous_H0_6822_baseline"]
    prof = g3["unconstrained_alpha_profile_check"]
    arch = dict(np.load(DATA_DIR / "cmb_planck_pr3_act_dr6_archival.npz"))

    plt.rcParams.update({"font.size": 9.2, "axes.titlesize": 9.8, "axes.labelsize": 9.2})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.3))
    fig.subplots_adjust(hspace=0.33, wspace=0.26, left=0.075, right=0.98, top=0.93, bottom=0.08)

    # Panel A: High-ell Silk Tail (ell in [900, 3000]) EE Polarization showing elimination of +0.271% phase shift
    ax = axes[0, 0]
    ells = np.array(cv["ells"])
    m_hi = ells >= 850
    ax.errorbar(
        arch["act_dr6_ell_ee"][arch["act_dr6_ell_ee"] >= 850],
        arch["act_dr6_val_ee"][arch["act_dr6_ell_ee"] >= 850],
        yerr=np.sqrt(np.diag(arch["act_dr6_cov_108"])[72:])[arch["act_dr6_ell_ee"] >= 850],
        fmt="o",
        ms=3.5,
        color="#d97706",
        ecolor="#f59e0b",
        elinewidth=1.0,
        label="ACT DR6 EE ($\\ell \\leq 3000$)",
    )
    ax.plot(
        ells[m_hi],
        np.array(cv["Dl_EE_6822_2F"])[m_hi],
        color="#dc2626",
        ls="--",
        lw=1.4,
        label=(
            f"Homogén $H_0=68{{,}}22$ ($\\chi^2={fmt_num(hom_68['CMB_2Field_COBE']['act_dr6_108_cal_chi2'], 1, lang)}$)"
            if lang == "hu"
            else f"Homog. $H_0=68.22$ ($\\chi^2={hom_68['CMB_2Field_COBE']['act_dr6_108_cal_chi2']:.1f}$)"
        ),
    )
    ax.plot(
        ells[m_hi],
        np.array(cv["Dl_EE_vc_2F"])[m_hi],
        color="#2563eb",
        lw=2.0,
        label=(
            f"Üreg-óra $\\alpha_{{\\mathrm{{OCSR}}}}$ ($\\chi^2/108={fmt_num(vcA['CMB_2Field_COBE']['act_dr6_108_cal_chi2_dof'], 3, lang)}$)"
            if lang == "hu"
            else f"Void-clock $\\alpha_{{\\mathrm{{OCSR}}}}$ ($\\chi^2/108={vcA['CMB_2Field_COBE']['act_dr6_108_cal_chi2_dof']:.3f}$)"
        ),
    )
    ax.set_xlim(850, 3000)
    ax.set_xlabel("Multipólus $\\ell$" if lang == "hu" else "Multipole $\\ell$")
    ax.set_ylabel(r"$D_\ell^{EE}$ [$\mu\mathrm{K}^2$]")
    ax.set_title(
        "A. Kis szögméretű EE Silk-farok ($\\ell \\in [850, 3000]$): $0{,}000\\%$ $\\theta_*$-eltolódás"
        if lang == "hu"
        else "A. Small-Scale EE Silk Tail ($\\ell \\in [850, 3000]$): $0.000\\%$ $\\theta_*$ Shift"
    )
    ax.legend(loc="upper right", fontsize=7.9)
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_comma_formatter(0))
        ax.yaxis.set_major_formatter(make_comma_formatter(0))

    # Panel B: 1D Unconstrained alpha Scan at theta*-locked H0(alpha)
    ax = axes[0, 1]
    al_s = np.array(cv["alpha_scan"])
    c_bao = np.array(cv["chi2_DESI_BAO_of_alpha"])
    dc_pan = np.array(cv["dchi2_Pantheon_of_alpha"])
    dc_des5 = np.array(cv["dchi2_DES5YR_of_alpha"])
    tot_pan = (c_bao - c_bao[0]) + dc_pan
    tot_des5 = (c_bao - c_bao[0]) + dc_des5

    ax.axhline(0.0, color="#64748b", ls=":", lw=1.0)
    ax.plot(
        al_s,
        c_bao - c_bao[0],
        color="#059669",
        lw=1.8,
        ls="--",
        label="DESI 13-pont BAO $\\Delta\\chi^2$" if lang == "hu" else "DESI 13-point BAO $\\Delta\\chi^2$",
    )
    ax.plot(al_s, tot_pan, color="#2563eb", lw=2.1, label="DESI BAO + Pantheon+ $\\Delta\\chi^2$")
    ax.plot(al_s, tot_des5, color="#7c3aed", lw=1.8, ls="-.", label="DESI BAO + DES-SN5YR $\\Delta\\chi^2$")
    ax.axvline(prof["alpha_OCSR_locked"], color="#dc2626", lw=1.9, label=f"Rögzített $\\alpha_{{\\mathrm{{OCSR}}}} = 5\\sqrt{{3}}/24$" if lang == "hu" else f"Locked $\\alpha_{{\\mathrm{{OCSR}}}} = 5\\sqrt{{3}}/24$")
    ax.axvspan(prof["alpha_best_BAO_plus_Pantheon"] - prof["sigma_alpha"], prof["alpha_best_BAO_plus_Pantheon"] + prof["sigma_alpha"], color="#93c5fd", alpha=0.25)
    ax.set_xlim(0.0, 0.68)
    ax.set_xlabel(r"Üreg-óra súlyparaméter $\alpha$ ($\theta_*$-rögzített $H_0(\alpha)$)" if lang == "hu" else r"Void-clock weight $\alpha$ ($\theta_*$-locked $H_0(\alpha)$)")
    ax.set_ylabel(r"$\Delta\chi^2$ a homogén $\theta_*$-alapvonalhoz képest" if lang == "hu" else r"$\Delta\chi^2$ vs. homogeneous $\theta_*$-locked baseline")
    ax.set_title(
        f"B. Zéró-paraméteres $\\alpha_{{\\mathrm{{OCSR}}}}$ egybeesés (${fmt_num(prof['z_offset_OCSR_vs_best'], 2, lang)}\\sigma$ az optimumtól)"
        if lang == "hu"
        else f"B. Zero-Parameter $\\alpha_{{\\mathrm{{OCSR}}}}$ Match (${prof['z_offset_OCSR_vs_best']:+.2f}\\sigma$ from Optimum)"
    )
    ax.legend(loc="upper right", fontsize=7.7)
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_comma_formatter(2))
        ax.yaxis.set_major_formatter(make_comma_formatter(0))

    # Panel C: DESI 13-pt BAO & ACT DR6 108-bin Simultaneous Tension Elimination
    ax = axes[1, 0]
    models_c = (
        ["Homogén\n$\\theta_*$-rögzített", "Homogén\n$H_0=68{,}22$", "Üreg-óra Ág A\n($k_{\\mathrm{added}}=0$)", "Üreg-óra Ág B\n($k_{\\mathrm{added}}=0$)"]
        if lang == "hu"
        else ["Homog.\n$\\theta_*$-locked", "Homog.\n$H_0=68.22$", "Void-Clock A\n($k_{\\mathrm{added}}=0$)", "Void-Clock B\n($k_{\\mathrm{added}}=0$)"]
    )
    x_c = np.arange(4)
    w_bar = 0.35
    act_dof = [
        hom_th["CMB_2Field_COBE"]["act_dr6_108_cal_chi2_dof"],
        hom_68["CMB_2Field_COBE"]["act_dr6_108_cal_chi2_dof"],
        vcA["CMB_2Field_COBE"]["act_dr6_108_cal_chi2_dof"],
        vcB["CMB_2Field_COBE"]["act_dr6_108_cal_chi2_dof"],
    ]
    bao_dof = [
        hom_th["late_probes"]["DESI_13_BAO_chi2_dof"],
        hom_68["late_probes"]["DESI_13_BAO_chi2_dof"],
        vcA["late_probes"]["DESI_13_BAO_chi2_dof"],
        vcB["late_probes"]["DESI_13_BAO_chi2_dof"],
    ]
    b1 = ax.bar(x_c - w_bar / 2, act_dof, w_bar, color="#2563eb", label="ACT DR6 (108 sáv, $\\ell \\leq 3000$)" if lang == "hu" else "ACT DR6 (108 bins, $\\ell \\leq 3000$)")
    b2 = ax.bar(x_c + w_bar / 2, bao_dof, w_bar, color="#10b981", label="DESI BAO (13 pont)" if lang == "hu" else "DESI BAO (13 points)")
    ax.axhline(1.0, color="#64748b", ls="--", lw=1.0)
    ax.set_xticks(x_c)
    ax.set_xticklabels(models_c, fontsize=8.0)
    ax.set_ylim(0.0, 3.05)
    ax.set_ylabel(r"$\chi^2 / \mathrm{dof}$")
    ax.set_title(
        "C. A $\\theta_*$–DESI BAO feszültség feloldása szabad paraméter nélkül"
        if lang == "hu"
        else "C. Simultaneous Elimination of $\\theta_*$--DESI BAO Tension ($k_{\\mathrm{added}}=0$)"
    )
    for rect_set in (b1, b2):
        for r_b in rect_set:
            hv = r_b.get_height()
            ax.text(r_b.get_x() + r_b.get_width() / 2.0, hv + 0.05, fmt_plain(hv, 2, lang), ha="center", fontsize=7.6, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.0)
    if lang == "hu":
        ax.yaxis.set_major_formatter(make_comma_formatter(1))

    # Panel D: Out-of-Sample SN Ia + CMB Lensing + CC15 Improvements at Zero Free Parameters
    ax = axes[1, 1]
    probes_d = ["Pantheon+\n(1580 SN)", "DES-Dovekie\n(1820 SN)", "DES-SN5YR\n(1829 SN)", "Union3\n(22 csomó)" if lang == "hu" else "Union3\n(22 nodes)", "DESI BAO\n(13 pont)" if lang == "hu" else "DESI BAO\n(13 pts)"]
    dchi_vs_hom_th = [
        vcA["late_probes"]["SN_chi2"]["PantheonPlus"] - hom_th["late_probes"]["SN_chi2"]["PantheonPlus"],
        vcA["late_probes"]["SN_chi2"]["DES_Dovekie"] - hom_th["late_probes"]["SN_chi2"]["DES_Dovekie"],
        vcA["late_probes"]["SN_chi2"]["DES_SN5YR_2024"] - hom_th["late_probes"]["SN_chi2"]["DES_SN5YR_2024"],
        vcA["late_probes"]["SN_chi2"]["Union3"] - hom_th["late_probes"]["SN_chi2"]["Union3"],
        vcA["late_probes"]["DESI_13_BAO_chi2"] - hom_th["late_probes"]["DESI_13_BAO_chi2"],
    ]
    x_d = np.arange(len(probes_d))
    bars_d = ax.bar(x_d, dchi_vs_hom_th, width=0.54, color="#0ea5e9", edgecolor="#0f172a", lw=0.8)
    ax.axhline(0.0, color="#334155", lw=1.0)
    ax.set_xticks(x_d)
    ax.set_xticklabels(probes_d, fontsize=8.0)
    ax.set_ylim(-26.0, 3.5)
    ax.set_ylabel(r"$\Delta\chi^2$ ($\alpha_{\mathrm{OCSR}}$ mínusz homogén $\theta_*$)" if lang == "hu" else r"$\Delta\chi^2$ ($\alpha_{\mathrm{OCSR}}$ minus homog. $\theta_*$)")
    ax.set_title(
        "D. Rögzített szótárú javulás mind az 5 késői adatsoron ($\\Delta k_{\\mathrm{add}} = 0$, D1: $a_0$ illesztett)"
        if lang == "hu"
        else "D. Locked-Dictionary $\\Delta\\chi^2 < 0$ Across All 5 Late-Time Probes ($\\Delta k_{\\mathrm{add}} = 0$, D1)"
    )
    for rb, vd in zip(bars_d, dchi_vs_hom_th):
        ax.text(rb.get_x() + rb.get_width() / 2.0, vd - 1.55, fmt_plain(vd, 2, lang), ha="center", fontsize=8.0, fontweight="bold", color="#0f172a")
    if lang == "hu":
        ax.yaxis.set_major_formatter(make_comma_formatter(0))

    out_name = "fig2_zero_param_theta_bao_sn_boltzmann_3000_hu.pdf" if lang == "hu" else "fig2_zero_param_theta_bao_sn_boltzmann_3000.pdf"
    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)


def main() -> None:
    res = json.loads(JSON_IN.read_text(encoding="utf-8"))
    for lg in ("en", "hu"):
        build_fig1(res, lg)
        build_fig2(res, lg)
    print("[Paper 29 - Step 1] Generated all 4 bilingual vector PDFs cleanly.")


if __name__ == "__main__":
    main()
