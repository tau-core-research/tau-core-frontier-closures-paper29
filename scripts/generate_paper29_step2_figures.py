#!/usr/bin/env python3
"""
JSON-Driven Vector PDF Figure Generator (EN + HU) for Tau Core Paper 29 — Step 2 (Pillar 2, v1.2.0-step2):
  - figures/fig3_bullet_cluster_3d_hydro_qumond_maps[_hu].pdf
  - figures/fig4_violent_relaxation_and_64_cluster_apertures[_hu].pdf

Strict Referee Formatting Rules:
  - Must pass `python3 -W error` cleanly with exit code 0.
  - 100% JSON-driven N-body & merger curves (zero duplicate N-body integrators).
  - 100% Hungarian decimal commas (`{,}` inside mathtext, `,` in plain text) and zero English leaks on `_hu.pdf`.
  - Uses `\\leq` / `\\geq` (never `\\le` / `\\ge`) for Matplotlib mathtext compatibility.
  - Wide subplot gutters (`wspace=0.40, hspace=0.36`) and clear legend margins to guarantee zero overlaps.
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
FIG_DIR = ROOT_DIR / "figures"
JSON_IN = ROOT_DIR / "paper29_step2_verification_results.json"


def fmt_num(val: float, dec: int, lang: str) -> str:
    s = f"{val:.{dec}f}"
    return s.replace(".", "{,}") if lang == "hu" else s


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


def make_log_comma_formatter():
    def _fmt(x: float, _pos: int) -> str:
        if x <= 0.0:
            return "0"
        exp = int(round(math.log10(x)))
        if abs(x - (10.0**exp)) < 1.0e-9 * (10.0**exp):
            return rf"$10^{{{exp}}}$"
        return f"{x:g}".replace(".", ",")
    return FuncFormatter(_fmt)


def build_fig3(res: Dict[str, Any], lang: str) -> None:
    g3 = res["gate_S2_3_live_3d_bullet_cluster_hydro_qumond_merger"]
    maps = g3["plot_maps_2d"]
    hyd = g3["hydro_and_xray_diagnostics"]
    mc = g3["model_comparison_on_live_3d_state"]

    xk = np.array(maps["x_kpc"], dtype=float)
    yk = np.array(maps["y_kpc"], dtype=float)
    em = np.array(maps["xray_em_norm"], dtype=float).T
    kap_m3 = np.array(maps["kappa_M3a"], dtype=float).T
    kap_m0 = np.array(maps["kappa_M0a"], dtype=float).T

    bcg_m = hyd["bcg_main_kpc"]
    bcg_s = hyd["bcg_sub_kpc"]
    xr_m = hyd["xray_peak_main_kpc"]
    xr_s = hyd["xray_peak_sub_kpc"]
    mach_v = hyd["bow_shock_mach_number_sim"]

    m0a = mc["M0a_baryon_only_3D_QUMOND_Sub1_NOGO"]
    m1k = mc["M1_naive_flat_King_cK_2p015_NOGO"]
    m2a = mc["M2a_Paper28_unscreened_psiIR_only"]
    m2b = mc["M2b_massive_gravity_total_Yukawa_NOGO"]
    m3a = mc["M3a_Paper28_AeST_live_median_c200"]
    m3b = mc["M3b_Paper28_AeST_live_plus1sigma_fixedR200"]
    m3c = mc["M3c_Paper28_AeST_live_Paraficz_c200_fixedR200"]

    plt.rcParams.update({"font.size": 9.0, "axes.titlesize": 9.6, "axes.labelsize": 9.0})
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.5))
    fig.subplots_adjust(hspace=0.36, wspace=0.40, left=0.066, right=0.938, top=0.93, bottom=0.085)

    # --- Panel A: Live 3D M3a X-ray EM + Weak-Lensing kappa contours ---
    ax = axes[0, 0]
    im_a = ax.imshow(
        np.sqrt(np.maximum(em, 0.0)),
        extent=[xk[0], xk[-1], yk[0], yk[-1]],
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    cb_a = fig.colorbar(im_a, ax=ax, fraction=0.044, pad=0.025)
    cb_a.set_label(r"$\sqrt{\mathrm{EM}/\mathrm{EM}_{\max}}$", fontsize=8.4)
    if lang == "hu":
        cb_a.ax.yaxis.set_major_formatter(make_comma_formatter(1))

    lev_m3 = np.linspace(0.06, float(np.max(kap_m3)) * 0.92, 6)
    ax.contour(xk, yk, kap_m3, levels=lev_m3, colors="#38bdf8", linewidths=1.35)
    ax.plot([bcg_m[0], bcg_s[0]], [bcg_m[1], bcg_s[1]], "*", ms=10.5, color="#fde047", mec="black", mew=0.7,
            label=("BCG centrumok" if lang == "hu" else "BCG centroids"))
    ax.plot([xr_m[0], xr_s[0]], [xr_m[1], xr_s[1]], "x", ms=8.0, mew=2.0, color="#ffffff",
            label=("Röntgen EM csúcsok" if lang == "hu" else "X-ray EM peaks"))
    lp3_m, lp3_s = m3a["lens_peak_main_kpc"], m3a["lens_peak_sub_kpc"]
    ax.plot([lp3_m[0], lp3_s[0]], [lp3_m[1], lp3_s[1]], "o", ms=6.8, mfc="none", mec="#38bdf8", mew=2.0,
            label=(r"$\kappa$ lencsecsúcsok (M3a)" if lang == "hu" else r"$\kappa$ lensing peaks (M3a)"))
    off_m_s = fmt_num(m3a["gas_to_lens_offset_main_kpc"], 1, lang)
    off_s_s = fmt_num(m3a["gas_to_lens_offset_sub_kpc"], 1, lang)
    ax.text(
        0.03,
        0.05,
        (
            rf"Ívhullám $\mathcal{{M}} = {fmt_num(mach_v, 2, lang)}$" + "\n"
            rf"Eltolás: $({off_m_s};\; {off_s_s})\;\mathrm{{kpc}}$"
            if lang == "hu"
            else rf"Bow shock $\mathcal{{M}} = {mach_v:.2f}$" + "\n"
            rf"Offsets: $({off_m_s}, {off_s_s})\;\mathrm{{kpc}}$"
        ),
        transform=ax.transAxes,
        fontsize=7.9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc="#0f172a", ec="#38bdf8", alpha=0.84),
    )
    ax.set_title(
        "A. Élő 3D AeST + QUMOND (M3a): röntgen gáz és elvált lencsézés"
        if lang == "hu"
        else "A. Live 3D AeST + QUMOND (M3a): X-Ray Gas vs. Separated Lensing"
    )
    ax.set_xlabel(r"$x\;\mathrm{[kpc]}$")
    ax.set_ylabel(r"$y\;\mathrm{[kpc]}$")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.82, facecolor="#0f172a", labelcolor="white")

    # --- Panel B: Counterfactual Baryon-Only 3D QUMOND M0a ---
    ax = axes[0, 1]
    im_b = ax.imshow(
        np.sqrt(np.maximum(em, 0.0)),
        extent=[xk[0], xk[-1], yk[0], yk[-1]],
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    cb_b = fig.colorbar(im_b, ax=ax, fraction=0.044, pad=0.025)
    cb_b.set_label(r"$\sqrt{\mathrm{EM}/\mathrm{EM}_{\max}}$", fontsize=8.4)
    if lang == "hu":
        cb_b.ax.yaxis.set_major_formatter(make_comma_formatter(1))

    lev_m0 = np.linspace(0.035, float(np.max(kap_m0)) * 0.92, 6)
    ax.contour(xk, yk, kap_m0, levels=lev_m0, colors="#4ade80", linewidths=1.35)
    ax.plot([bcg_m[0], bcg_s[0]], [bcg_m[1], bcg_s[1]], "*", ms=10.5, color="#fde047", mec="black", mew=0.7,
            label=("BCG centrumok" if lang == "hu" else "BCG centroids"))
    ax.plot([xr_m[0], xr_s[0]], [xr_m[1], xr_s[1]], "x", ms=8.0, mew=2.0, color="#ffffff",
            label=("Röntgen EM csúcsok" if lang == "hu" else "X-ray EM peaks"))
    lp0_m, lp0_s = m0a["lens_peak_main_kpc"], m0a["lens_peak_sub_kpc"]
    ax.plot([lp0_m[0], lp0_s[0]], [lp0_m[1], lp0_s[1]], "s", ms=6.6, mfc="none", mec="#4ade80", mew=2.0,
            label=(r"$\kappa$ csúcsok: csak-barion QUMOND" if lang == "hu" else r"$\kappa$ peaks: Baryon-Only QUMOND"))
    ax.text(
        0.03,
        0.05,
        (
            rf"KIZÁRVA ($\rho_A = 0$):" + "\n"
            rf"Bullet eltolás $= {fmt_num(m0a['gas_to_lens_offset_sub_kpc'], 1, lang)}\;\mathrm{{kpc}}$ (${fmt_num(m0a['z_offset_sub_vs_Clowe2006'], 2, lang)}\sigma$)"
            if lang == "hu"
            else rf"PROVEN NO-GO ($\rho_A = 0$):" + "\n"
            rf"Bullet offset $= {m0a['gas_to_lens_offset_sub_kpc']:.1f}\;\mathrm{{kpc}}$ (${m0a['z_offset_sub_vs_Clowe2006']:+.2f}\sigma$)"
        ),
        transform=ax.transAxes,
        fontsize=7.9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc="#450a0a", ec="#f87171", alpha=0.88),
    )
    ax.set_title(
        "B. Csak-barion 3D QUMOND (M0a): lencsézés a gázhoz tapad"
        if lang == "hu"
        else "B. Baryon-Only 3D QUMOND (M0a): Lensing Pinned to Gas (NO-GO)"
    )
    ax.set_xlabel(r"$x\;\mathrm{[kpc]}$")
    ax.set_ylabel(r"$y\;\mathrm{[kpc]}$")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.82, facecolor="#0f172a", labelcolor="white")

    # --- Panel C: Gas-to-Lensing Spatial Offsets (Main SE & Bullet NW) ---
    ax = axes[1, 0]
    models_c = (
        ["M0a\nCsak-bar.", "M1\n1 Mpc King", "M2b\nYukawa-G", "M3a\nMedián $c_{200}$", "M3b\nÉlő $+1\\sigma$", "M3c\nParaficz+"]
        if lang == "hu"
        else ["M0a\nBar.-Only", "M1\n1-Mpc King", "M2b\nYukawa-G", "M3a\nMedian $c_{200}$", "M3b\nLive $+1\\sigma$", "M3c\nParaficz+"]
    )
    off_main_vals = [
        m0a["gas_to_lens_offset_main_kpc"],
        m1k["gas_to_lens_offset_main_kpc"],
        m2b["gas_to_lens_offset_main_kpc"],
        m3a["gas_to_lens_offset_main_kpc"],
        m3b["gas_to_lens_offset_main_kpc"],
        m3c["gas_to_lens_offset_main_kpc"],
    ]
    off_sub_vals = [
        m0a["gas_to_lens_offset_sub_kpc"],
        m1k["gas_to_lens_offset_sub_kpc"],
        m2b["gas_to_lens_offset_sub_kpc"],
        m3a["gas_to_lens_offset_sub_kpc"],
        m3b["gas_to_lens_offset_sub_kpc"],
        m3c["gas_to_lens_offset_sub_kpc"],
    ]
    xp = np.arange(len(models_c))
    w = 0.35
    ax.axhspan(185.0 - 25.0, 185.0 + 25.0, color="#2563eb", alpha=0.13,
               label=("Clowe+06 főhalmaz ($185\\pm 25$)" if lang == "hu" else "Clowe+06 Main ($185\\pm 25$)"))
    ax.axhspan(220.0 - 25.0, 220.0 + 25.0, color="#dc2626", alpha=0.13,
               label=("Clowe+06 Bullet ($220\\pm 25$)" if lang == "hu" else "Clowe+06 Bullet ($220\\pm 25$)"))
    ax.bar(xp - 0.5 * w, off_main_vals, width=w, color="#2563eb", alpha=0.86,
           label=("Főhalmaz eltolás" if lang == "hu" else "Main cluster offset"))
    ax.bar(xp + 0.5 * w, off_sub_vals, width=w, color="#dc2626", alpha=0.86,
           label=("Bullet alhalmaz eltolás" if lang == "hu" else "Bullet subcluster offset"))
    ax.set_xticks(xp)
    ax.set_xticklabels(models_c, fontsize=7.6)
    ax.set_ylim(0.0, 315.0)
    ax.set_ylabel("Gáz–lencsézés eltolás [kpc]" if lang == "hu" else "Gas-to-Lensing Offset [kpc]")
    ax.set_title(
        "C. Kétkomponensű térbeli gáz–lencsézés eltolások (Clowe+ 2006)"
        if lang == "hu"
        else "C. Dual Spatial Gas-to-Lensing Offsets vs. Clowe+ (2006)"
    )
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(loc="upper left", fontsize=7.2, ncol=2)

    # --- Panel D: Cylindrical R < 250 kpc Aperture Masses vs Bradac+ (2006) ---
    ax = axes[1, 1]
    models_d = (
        ["M0a\nCsak-b.", "M1\nKing", "M2a\nCsak-$\\psi$", "M2b\nYuk.-G", "M3a\nMedián", "M3b\nÉlő $+1\\sigma$", "M3c\nParaficz+"]
        if lang == "hu"
        else ["M0a\nBar.", "M1\nKing", "M2a\n$\\psi$-Only", "M2b\nYuk.-G", "M3a\nMedian", "M3b\nLive $+1\\sigma$", "M3c\nParaficz+"]
    )
    map_m_vals = [
        m0a["M_ap_250kpc_main_1e14Msun"],
        m1k["M_ap_250kpc_main_1e14Msun"],
        m2a["M_ap_250kpc_main_1e14Msun"],
        m2b["M_ap_250kpc_main_1e14Msun"],
        m3a["M_ap_250kpc_main_1e14Msun"],
        m3b["M_ap_250kpc_main_1e14Msun"],
        m3c["M_ap_250kpc_main_1e14Msun"],
    ]
    map_s_vals = [
        m0a["M_ap_250kpc_sub_1e14Msun"],
        m1k["M_ap_250kpc_sub_1e14Msun"],
        m2a["M_ap_250kpc_sub_1e14Msun"],
        m2b["M_ap_250kpc_sub_1e14Msun"],
        m3a["M_ap_250kpc_sub_1e14Msun"],
        m3b["M_ap_250kpc_sub_1e14Msun"],
        m3c["M_ap_250kpc_sub_1e14Msun"],
    ]
    xd = np.arange(len(models_d))
    ax.axhspan(2.80 - 0.30, 2.80 + 0.30, color="#1d4ed8", alpha=0.13,
               label=(r"Bradač+06 fő ($2{,}80\pm 0{,}30$)" if lang == "hu" else r"Bradač+06 Main ($2.80\pm 0.30$)"))
    ax.axhspan(1.15 - 0.20, 1.15 + 0.20, color="#059669", alpha=0.14,
               label=(r"Bradač+06 Bullet ($1{,}15\pm 0{,}20$)" if lang == "hu" else r"Bradač+06 Bullet ($1.15\pm 0.20$)"))
    ax.bar(xd - 0.5 * w, map_m_vals, width=w, color="#1d4ed8", alpha=0.86,
           label=("Főhalmaz $M_{\\mathrm{ap}}(<250)$" if lang == "hu" else "Main $M_{\\mathrm{ap}}(<250)$"))
    ax.bar(xd + 0.5 * w, map_s_vals, width=w, color="#059669", alpha=0.86,
           label=("Bullet $M_{\\mathrm{ap}}(<250)$" if lang == "hu" else "Bullet $M_{\\mathrm{ap}}(<250)$"))
    ax.set_xticks(xd)
    ax.set_xticklabels(models_d, fontsize=7.2)
    ax.set_ylim(0.0, 3.55)
    if lang == "hu":
        ax.yaxis.set_major_formatter(make_comma_formatter(1))
    ax.set_ylabel(r"$M_{\mathrm{ap}}(<250\;\mathrm{kpc})\;[10^{14}\,M_\odot]$")
    ax.set_title(
        "D. Hengeres $250\\;\\mathrm{kpc}$ lencsézési tömegek (Bradač+ 2006)"
        if lang == "hu"
        else "D. Cylindrical $250\\;\\mathrm{kpc}$ Aperture Masses vs. Bradač+ (2006)"
    )
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(loc="upper left", fontsize=7.1, ncol=2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_p = FIG_DIR / ("fig3_bullet_cluster_3d_hydro_qumond_maps_hu.pdf" if lang == "hu" else "fig3_bullet_cluster_3d_hydro_qumond_maps.pdf")
    fig.savefig(out_p)
    plt.close(fig)


def build_fig4(res: Dict[str, Any], lang: str) -> None:
    g1 = res["gate_S2_1_director_expulsion_noether_and_neutrino_nogo"]
    g2 = res["gate_S2_2_symplectic_violent_relaxation_core_formation"]
    g4 = res["gate_S2_4_xcop_clash_64_aperture_multicluster_audit"]
    lc = res["locked_constants"]

    plt.rcParams.update({"font.size": 9.0, "axes.titlesize": 9.6, "axes.labelsize": 9.0})
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.5))
    fig.subplots_adjust(hspace=0.36, wspace=0.38, left=0.072, right=0.985, top=0.93, bottom=0.085)

    # --- Panel A: Gate S2.1 3+1D Director Expulsion Profile m_H,loc(r)/m_H,inf ---
    ax = axes[0, 0]
    r_kpc = np.geomspace(15.0, 40000.0, 400)
    rs_cl = 520.0
    env_250 = math.exp(-((250.0 / 6000.0) ** 2))
    phi0_c2 = g1["cluster_well_phi_over_c2"] / ((math.log(1.0 + 250.0 / rs_cl) / (250.0 / rs_cl)) * env_250)
    u_r = r_kpc / rs_cl
    phi_c2 = phi0_c2 * (np.log(1.0 + u_r) / u_r) * np.exp(-((r_kpc / 6000.0) ** 2))
    psi_c2 = 0.25 * phi_c2
    c2_dir = lc["c2_director_stiffness"]
    kap_th = lc["kappa_theta"]
    m_ratio = c2_dir / (c2_dir + kap_th * (psi_c2 * psi_c2 + psi_c2 * phi_c2))

    ax.loglog(
        r_kpc / 1000.0,
        m_ratio,
        color="#2563eb",
        lw=2.2,
        label=(r"$m_{H,\mathrm{loc}}/m_{H,\infty} = c_2 / [c_2 + \kappa_\theta(\psi_{\mathrm{IR}}^2 + \psi_{\mathrm{IR}}\phi_N)]$"),
    )
    ax.axhline(1.0, color="#64748b", ls="--", lw=1.0,
               label=("Árnyékolatlan lineáris kozmikus határ" if lang == "hu" else "Unscreened linear cosmic limit"))
    ax.scatter([0.25], [g1["director_expulsion_ratio_mHloc_over_mHinf"]], color="#dc2626", s=45, zorder=5)
    ax.text(
        0.03,
        0.54,
        (
            rf"Halmazmag ($250\;\mathrm{{kpc}}$): $m_{{H,\mathrm{{loc}}}}/m_{{H,\infty}} = {fmt_sci(g1['director_expulsion_ratio_mHloc_over_mHinf'], 2, lang)}$" + "\n"
            rf"Izolált $L_*$ $\psi_{{\mathrm{{IR}}}}$ ($1\;\mathrm{{Mpc}}$): $V_c = {fmt_num(g1['mistele_Lstar_Vc_1Mpc_director_expelled_km_s'], 1, lang)}$ (árny.-lan: ${fmt_num(g1['mistele_Lstar_Vc_1Mpc_unscreened_NOGO_km_s'], 1, lang)}$) $\mathrm{{km/s}}$" + "\n"
            rf"Pauli Tremaine–Gunn ($\sum m_\nu = 0{{,}}059\;\mathrm{{eV}}$): ${fmt_sci(g1['tremaine_gunn_pauli_waterbag_shortfall_factor'], 2, lang)}\times$ hiány"
            if lang == "hu"
            else rf"Cluster core ($250\;\mathrm{{kpc}}$): $m_{{H,\mathrm{{loc}}}}/m_{{H,\infty}} = {fmt_sci(g1['director_expulsion_ratio_mHloc_over_mHinf'], 2, lang)}$" + "\n"
            rf"Isolated $L_*$ $\psi_{{\mathrm{{IR}}}}$ ($1\;\mathrm{{Mpc}}$): $V_c = {g1['mistele_Lstar_Vc_1Mpc_director_expelled_km_s']:.1f}$ vs. ${g1['mistele_Lstar_Vc_1Mpc_unscreened_NOGO_km_s']:.1f}\;\mathrm{{km/s}}$" + "\n"
            rf"Pauli Tremaine–Gunn ($\sum m_\nu = 0.059\;\mathrm{{eV}}$): ${fmt_sci(g1['tremaine_gunn_pauli_waterbag_shortfall_factor'], 2, lang)}\times$ shortfall"
        ),
        transform=ax.transAxes,
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.26", fc="#f8fafc", ec="#cbd5e1"),
    )
    ax.set_xlabel(r"$r\;\mathrm{[Mpc]}$")
    ax.set_ylabel(r"$m_{H,\mathrm{loc}}(r) / m_{H,\infty}$")
    # Lower ylim to 5e-10 so the legend at lower right sits cleanly below the curve with zero intersection!
    ax.set_ylim(5.0e-10, 3.0)
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_log_comma_formatter())
        ax.yaxis.set_major_formatter(make_log_comma_formatter())
    ax.set_title(
        "A. Kapu S2.1: 3+1D direktor-kiszorítás és neutrínó kizárás"
        if lang == "hu"
        else "A. Gate S2.1: 3+1D Director Expulsion & Neutrino NO-GO"
    )
    ax.grid(True, which="both", alpha=0.22, ls=":")
    ax.legend(loc="lower right", fontsize=7.0, framealpha=0.94)

    # --- Panel B: Gate S2.2 100% JSON-Driven 3D Symplectic Violent-Relaxation Core vs Cold Cusp ---
    ax = axes[0, 1]
    pre = g2["preheated_condensate_nbody"]
    cld = g2["cold_dust_control_nbody"]
    r_pre = np.array(pre["r_over_rs_bins"], dtype=float)
    rho_pre = np.array(pre["rho_bins"], dtype=float)
    r_cld = np.array(cld["r_over_rs_bins"], dtype=float)
    rho_cld = np.array(cld["rho_bins"], dtype=float)

    ax.loglog(
        r_cld,
        rho_cld / rho_pre[0],
        "s--",
        color="#dc2626",
        ms=4.5,
        lw=1.6,
        label=(
            rf"Hideg por ($\sigma_0/V = 0{{,}}040$): meredekség $= {fmt_num(cld['inner_core_log_density_slope'], 2, lang)}$"
            if lang == "hu"
            else rf"Cold dust ($\sigma_0/V = 0.040$): slope $= {cld['inner_core_log_density_slope']:.2f}$"
        ),
    )
    ax.loglog(
        r_pre,
        rho_pre / rho_pre[0],
        "o-",
        color="#2563eb",
        ms=4.8,
        lw=2.0,
        label=(
            rf"Előfűtött AeST ($\sqrt{{c_{{\mathrm{{mix}}}}/3}} = 0{{,}}241$): meredekség $= {fmt_num(pre['inner_core_log_density_slope'], 2, lang)}$"
            if lang == "hu"
            else rf"Pre-heated AeST ($\sqrt{{c_{{\mathrm{{mix}}}}/3}} = 0.241$): slope $= {pre['inner_core_log_density_slope']:.2f}$"
        ),
    )
    bc_fit = lc["c_mix"]
    prof_th = ((bc_fit + r_pre[0]) * ((1.0 + r_pre[0]) ** 2)) / ((bc_fit + r_pre) * ((1.0 + r_pre) ** 2))
    ax.loglog(
        r_pre,
        prof_th,
        color="#0f172a",
        ls=":",
        lw=1.7,
        label=(
            rf"Analitikus magos NFW ($b_{{\mathrm{{core}}}} = c_{{\mathrm{{mix}}}} = {fmt_num(bc_fit, 3, lang)}$)"
            if lang == "hu"
            else rf"Analytic cored NFW ($b_{{\mathrm{{core}}}} = c_{{\mathrm{{mix}}}} = {bc_fit:.3f}$)"
        ),
    )
    ax.set_xlabel(r"$r / r_s$")
    ax.set_ylabel(r"$\rho(r) / \rho_{\mathrm{core}}$")
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_log_comma_formatter())
        ax.yaxis.set_major_formatter(make_log_comma_formatter())
    ax.set_title(
        "B. Kapu S2.2: 3D szimplektikus erőszakos relaxáció ($t = 8{,}0$)"
        if lang == "hu"
        else "B. Gate S2.2: 3D Symplectic Violent Relaxation ($t = 8.0$)"
    )
    ax.grid(True, which="both", alpha=0.22, ls=":")
    ax.legend(loc="lower left", fontsize=7.3)

    # --- Panel C: 100% JSON-Driven 20 CLASH BCG Inner Strong-Lensing Cores (Global-R200 & h-Harmonized) ---
    ax = axes[1, 0]
    rb_list = g4["cored_cmix_condensate_EquivalencePrinciple"]["per_cluster_R_BCG_kpc"]
    rat_cor = g4["cored_cmix_condensate_EquivalencePrinciple"]["per_cluster_BCG_Mpred_Mobs"]
    rat_csp = g4["cuspy_NFW_condensate_b0_BCG_NOGO"]["per_cluster_BCG_Mpred_Mobs"]

    ax.axhline(1.0, color="#0f172a", ls="--", lw=1.2)
    ax.scatter(
        rb_list,
        rat_csp,
        marker="^",
        s=38,
        color="#dc2626",
        alpha=0.82,
        label=(
            rf"Csúcsos NFW ($b=0$): $\chi^2/20 = {fmt_num(g4['cuspy_NFW_condensate_b0_BCG_NOGO']['CLASH_20_BCG_chi2_dof'], 2, lang)}$"
            if lang == "hu"
            else rf"Cuspy NFW ($b=0$): $\chi^2/20 = {g4['cuspy_NFW_condensate_b0_BCG_NOGO']['CLASH_20_BCG_chi2_dof']:.2f}$"
        ),
    )
    ax.scatter(
        rb_list,
        rat_cor,
        marker="o",
        s=40,
        color="#2563eb",
        alpha=0.90,
        label=(
            rf"Magos AeST ($b=c_{{\mathrm{{mix}}}}$): $\chi^2/20 = {fmt_num(g4['cored_cmix_condensate_EquivalencePrinciple']['CLASH_20_BCG_chi2_dof'], 2, lang)}$"
            if lang == "hu"
            else rf"Cored AeST ($b=c_{{\mathrm{{mix}}}}$): $\chi^2/20 = {g4['cored_cmix_condensate_EquivalencePrinciple']['CLASH_20_BCG_chi2_dof']:.2f}$"
        ),
    )
    ax.set_xlabel(r"$R_{\mathrm{BCG}}\;\mathrm{[kpc]}$")
    ax.set_ylabel(r"$M_{\mathrm{pred}}(<R_{\mathrm{BCG}}) / M_{\mathrm{lens,obs}}$")
    ax.set_ylim(0.45, 2.98)
    if lang == "hu":
        ax.xaxis.set_major_formatter(make_comma_formatter(1))
        ax.yaxis.set_major_formatter(make_comma_formatter(2))
    ax.set_title(
        "C. 20 CLASH BCG erős-lencsézési mag ($14\\text{–}32\\;\\mathrm{kpc}$, Tian+ 2020)"
        if lang == "hu"
        else "C. 20 CLASH BCG Strong-Lensing Cores ($14\\text{–}32\\;\\mathrm{kpc}$, Tian+ 2020)"
    )
    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(loc="upper right", fontsize=7.5)

    # --- Panel D: 64-Aperture Multi-Scale Summary ---
    ax = axes[1, 1]
    vb = g4["baryon_only_QUMOND_NOGO"]
    vy = g4["massive_gravity_total_Yukawa_NOGO"]
    vc = g4["cuspy_NFW_condensate_b0_BCG_NOGO"]
    ve = g4["cored_cmix_condensate_EquivalencePrinciple"]
    vw = g4["cored_cmix_condensate_Gate20_wGC"]

    cats = (
        ["Csak-barion\nQUMOND", "Yukawa-G\n$m_{H,\\infty}$", "Csúcsos\n$b=0$ NFW", "Magos $c_{\\mathrm{mix}}$\n(EP $s_{\\mathrm{tot}}$)", "Magos $c_{\\mathrm{mix}}$\n($w_{\\mathrm{GC}}$)"]
        if lang == "hu"
        else ["Baryon-Only\nQUMOND", "Yukawa-G\n$m_{H,\\infty}$", "Cuspy\n$b=0$ NFW", "Cored $c_{\\mathrm{mix}}$\n(EP $s_{\\mathrm{tot}}$)", "Cored $c_{\\mathrm{mix}}$\n($w_{\\mathrm{GC}}$)"]
    )
    xc_v = [vb["XCOP_24_chi2_dof"], vy["XCOP_24_chi2_dof"], vc["XCOP_24_chi2_dof"], ve["XCOP_24_chi2_dof"], vw["XCOP_24_chi2_dof"]]
    c25_v = [vb["CLASH_20_r2500_chi2_dof"], vy["CLASH_20_r2500_chi2_dof"], vc["CLASH_20_r2500_chi2_dof"], ve["CLASH_20_r2500_chi2_dof"], vw["CLASH_20_r2500_chi2_dof"]]
    bcg_v = [vb["CLASH_20_BCG_chi2_dof"], vy["CLASH_20_BCG_chi2_dof"], vc["CLASH_20_BCG_chi2_dof"], ve["CLASH_20_BCG_chi2_dof"], vw["CLASH_20_BCG_chi2_dof"]]

    xg = np.arange(len(cats))
    bw = 0.25
    ax.bar(xg - bw, xc_v, width=bw, color="#1d4ed8", label=("X-COP 24 ($R_{500},R_{200}$)"))
    ax.bar(xg, c25_v, width=bw, color="#059669", label=("CLASH 20 ($r_{2500}^X$)"))
    ax.bar(xg + bw, bcg_v, width=bw, color="#d97706", label=("CLASH 20 (BCG mag)" if lang == "hu" else "CLASH 20 (BCG core)"))
    ax.axhline(1.0, color="#64748b", ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 130.0)
    ax.set_xticks(xg)
    ax.set_xticklabels(cats, fontsize=7.6)
    if lang == "hu":
        ax.yaxis.set_major_formatter(make_log_comma_formatter())
    ax.set_ylabel(r"$\chi^2 / \mathrm{dof}$")
    ax.set_title(
        "D. Kapu S2.4: 64 valódi halmaz-apertúra ($14\\;\\mathrm{kpc} \\rightarrow 2{,}2\\;\\mathrm{Mpc}$)"
        if lang == "hu"
        else "D. Gate S2.4: 64 Real Cluster Apertures ($14\\;\\mathrm{kpc} \\rightarrow 2.2\\;\\mathrm{Mpc}$)"
    )
    ax.grid(axis="y", which="both", alpha=0.25, ls=":")
    ax.legend(loc="upper right", fontsize=7.3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_p = FIG_DIR / ("fig4_violent_relaxation_and_64_cluster_apertures_hu.pdf" if lang == "hu" else "fig4_violent_relaxation_and_64_cluster_apertures.pdf")
    fig.savefig(out_p)
    plt.close(fig)


def main() -> None:
    res = json.loads(JSON_IN.read_text(encoding="utf-8"))
    for lang in ("en", "hu"):
        build_fig3(res, lang)
        build_fig4(res, lang)
    print("[Generated Round-2 Step-2 EN + HU vector PDFs: fig3 & fig4]")


if __name__ == "__main__":
    main()
