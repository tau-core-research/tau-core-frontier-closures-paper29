#!/usr/bin/env python3
"""
Generate publication-grade bilingual (EN / HU) 4-panel figures for Paper 29 (v3.5.0):
  figures/fig_continuum_limit_a_to_0_en.pdf
  figures/fig_continuum_limit_a_to_0_hu.pdf

Reads directly from Stage-A verified JSON outputs (Modules H, I, J).
Zero hardcoded machine-specific paths; passes `python3 -W error` cleanly.
"""

from __future__ import annotations

import json
import pathlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
FIG_DIR = SCRIPT_DIR.parent / "figures"


def hu_fmt(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}".replace(".", ",")


def build_figure(lang: str, mod_h: dict, mod_i: dict, mod_j: dict, out_path: pathlib.Path) -> None:
    is_hu = lang == "hu"
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), constrained_layout=True)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # ------------------------------------------------------------------
    # Panel (a): Module H — 4D Spin(10) Quenched Pfaffian Phase & Plaquette Crossover
    # ------------------------------------------------------------------
    bscan = mod_h["gate2_3_dynamical_spin10_and_continuum_scaling"]["quenched_continuum_scaling_beta_scan"]
    betas = np.array([row["beta_gauge"] for row in bscan], dtype=float)
    rms_th = np.array([row["rms_theta_pf_rad"] for row in bscan], dtype=float)
    plaq = np.array([row["mean_plaquette"] for row in bscan], dtype=float)

    c1 = "#1b4f72"
    c2 = "#b03a2e"
    l1 = ax_a.loglog(
        betas,
        rms_th,
        "o-",
        color=c1,
        lw=2.0,
        ms=6.0,
        label=(
            r"Kioltott $\mathrm{RMS}(\theta_{\mathrm{Pf}})$ [rad]"
            if is_hu
            else r"Quenched $\mathrm{RMS}(\theta_{\mathrm{Pf}})$ [rad]"
        ),
    )
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(betas)
    ax_a.set_xticklabels([str(int(b)) for b in betas])
    ax_a.set_ylim(1.0e-4, 2.5e-2)
    ax_a.grid(True, which="both", ls=":", alpha=0.45)

    ax_a2 = ax_a.twinx()
    l2 = ax_a2.plot(
        betas,
        plaq,
        "s--",
        color=c2,
        lw=1.7,
        ms=5.2,
        label=(
            r"Átlagos plakett $\langle P_{\triangle}\rangle$"
            if is_hu
            else r"Mean plaquette $\langle P_{\triangle}\rangle$"
        ),
    )
    ax_a2.set_ylim(-0.12, 1.12)
    if is_hu:
        ax_a2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}".replace(".", ",")))

    ax_a.axvspan(16.0, 32.0, color="#f5b041", alpha=0.16)
    ax_a.set_xlabel(r"$\mathrm{Spin}(10)$ $\beta_{\mathrm{gauge}}$ ($g_0^2 \propto 1/\beta \to 0$)")
    ax_a.set_ylabel(r"$\mathrm{RMS}(\theta_{\mathrm{Pf}})$ [rad]", color=c1)
    ax_a2.set_ylabel(r"$\langle P_{\triangle}\rangle$", color=c2)
    ax_a.set_title(
        r"(a) H modul: 4D $\mathrm{Spin}(10)$ Pfaffian-fázis és átmenet ($a\to 0$)"
        if is_hu
        else r"(a) Module H: 4D $\mathrm{Spin}(10)$ Pfaffian Phase & Crossover ($a\to 0$)",
        fontweight="bold",
    )

    lines_a = l1 + l2
    labels_a = [ln.get_label() for ln in lines_a]
    leg_a = ax_a2.legend(lines_a, labels_a, loc="center left", bbox_to_anchor=(0.02, 0.36), framealpha=0.95)
    leg_a.set_zorder(10)

    note_a = (
        "Mérték-inv. $|\\Delta\\theta_{\\mathrm{Pf}}|=2{,}1\\times 10^{-15}$\n"
        "$b_0^{\\mathrm{GUT}}=76/3$; $24{,}2\\times$ elnyomás ($32\\to 128$)\n"
        "$L_s=6$ szivárgás: $2{,}7\\times 10^{-5}$ ($\\beta=32$)"
        if is_hu
        else "Gauge inv. $|\\Delta\\theta_{\\mathrm{Pf}}|=2.1\\times 10^{-15}$\n"
        "$b_0^{\\mathrm{GUT}}=76/3$; $24.2\\times$ drop ($32\\to 128$)\n"
        "$L_s=6$ leakage: $2.7\\times 10^{-5}$ ($\\beta=32$)"
    )
    ax_a.text(
        0.03,
        0.96,
        note_a,
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    # ------------------------------------------------------------------
    # Panel (b): Module I — 24-Cell O(a^2) Convergence & O(a^4) vs Z^4 Isotropy
    # ------------------------------------------------------------------
    ref_rows = mod_i["gate2b_continuum_refinement_and_isotropy"]["fixed_p_continuum_refinement"]
    iso_rows = mod_i["gate2b_continuum_refinement_and_isotropy"]["rotational_isotropy_d4_vs_z4"]

    N_grid = np.array([r["N_grid"] for r in iso_rows], dtype=float)
    err_tt = np.array([r["rel_err"] for r in ref_rows], dtype=float)
    aniso_d4 = np.array([r["d4_tt_rotational_anisotropy"] for r in iso_rows], dtype=float)
    aniso_z4 = np.array([r["z4_tt_rotational_anisotropy"] for r in iso_rows], dtype=float)

    ax_b.loglog(
        N_grid,
        err_tt,
        "^-",
        color="#117864",
        lw=2.0,
        ms=6.0,
        label=(
            r"$D_4$ rögzített-$p$ TT hiba ($\mathcal{O}(a^2)$, $4{,}02\times$)"
            if is_hu
            else r"$D_4$ fixed-$p$ TT rel. err ($\mathcal{O}(a^2)$, $4.02\times$)"
        ),
    )
    ax_b.loglog(
        N_grid,
        aniso_z4,
        "s--",
        color="#cb4335",
        lw=1.9,
        ms=5.6,
        label=(
            r"Hiperkockás $\mathbb{Z}^4$ forgási anizotrópia ($\mathcal{O}(a^2)$)"
            if is_hu
            else r"Hypercubic $\mathbb{Z}^4$ rot. anisotropy ($\mathcal{O}(a^2)$)"
        ),
    )
    ax_b.loglog(
        N_grid,
        aniso_d4,
        "o-",
        color="#1f618d",
        lw=2.2,
        ms=6.2,
        label=(
            r"24-cellás $D_4$ forgási anizotrópia ($\mathcal{O}(a^4)$, $16{,}00\times$)"
            if is_hu
            else r"24-cell $D_4$ rot. anisotropy ($\mathcal{O}(a^4)$, $16.00\times$)"
        ),
    )
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(N_grid)
    ax_b.set_xticklabels([str(int(n)) for n in N_grid])
    ax_b.set_ylim(3.0e-9, 1.0)
    ax_b.grid(True, which="both", ls=":", alpha=0.45)
    ax_b.set_xlabel(
        r"Rácsfinomítás $N_{\mathrm{grid}} \propto 1/a$ ($a\to 0$)"
        if is_hu
        else r"Lattice refinement $N_{\mathrm{grid}} \propto 1/a$ ($a\to 0$)"
    )
    ax_b.set_ylabel(
        r"Relatív hiba / forgási anizotrópia"
        if is_hu
        else r"Relative error / rotational anisotropy"
    )
    ax_b.set_title(
        r"(b) I modul: 24-cellás $\mathcal{O}(a^4)$ spin-2 izotrópia vs. $\mathbb{Z}^4$"
        if is_hu
        else r"(b) Module I: 24-Cell $\mathcal{O}(a^4)$ Spin-2 Isotropy vs. $\mathbb{Z}^4$",
        fontweight="bold",
    )
    ax_b.legend(loc="lower left", framealpha=0.92)

    note_b = (
        "$N=32$: $D_4$ $9\\,100\\times$ izotrópabb ($1{,}01\\times 10^{-8}$)\n"
        "Richardson TT hiba: $2{,}18\\times 10^{-5}$; $W(F_4)$ hiba: $0{,}0$"
        if is_hu
        else "$N=32$: $D_4$ is $9{,}100\\times$ more isotropic ($1.01\\times 10^{-8}$)\n"
        "Richardson TT err: $2.18\\times 10^{-5}$; $W(F_4)$ err: $0.0$"
    )
    ax_b.text(
        0.97,
        0.96,
        note_b,
        transform=ax_b.transAxes,
        va="top",
        ha="right",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    # ------------------------------------------------------------------
    # Panel (c): Module I — 4D D_4 Link-Length MCMC c_DW^{eff} & d_S^{norm}
    # ------------------------------------------------------------------
    mscan = mod_i["gate1b_and_2c_4d_link_mcmc_and_spectral_dim"]["gaussian_pinned_4d_d4_link_mcmc_scan"]
    g2_vals = np.array([r["g_grav_sq"] for r in mscan], dtype=float)
    cdw_vals = np.array([r["c_DW_eff"] for r in mscan], dtype=float)
    cdw_err = np.array([r["c_DW_eff_jackknife_stderr"] for r in mscan], dtype=float)
    ds_norm = np.array([r["spectral_dimension_torus_normalized_4d"] for r in mscan], dtype=float)

    c_dw_col = "#6c3483"
    c_ds_col = "#1e8449"
    ax_c.axhline(
        -1.0,
        color="#555555",
        ls="--",
        lw=1.4,
        label=r"24-cellás egzakt $c_{\mathrm{DW}}=-1$" if is_hu else r"24-cell exact $c_{\mathrm{DW}}=-1$",
    )
    eb = ax_c.errorbar(
        g2_vals,
        cdw_vals,
        yerr=cdw_err,
        fmt="o-",
        color=c_dw_col,
        lw=2.0,
        ms=5.8,
        capsize=3.5,
        label=(
            r"MCMC $c_{\mathrm{DW}}^{\mathrm{eff}}$ (jackknife $\pm 1\sigma$)"
            if is_hu
            else r"MCMC $c_{\mathrm{DW}}^{\mathrm{eff}}$ (jackknife $\pm 1\sigma$)"
        ),
    )
    ax_c.set_xscale("log")
    ax_c.invert_xaxis()  # Left -> Right corresponds to g_grav^2 -> 0 (continuum weak coupling!)
    ax_c.set_ylim(-1.55, -0.35)
    ax_c.grid(True, which="both", ls=":", alpha=0.45)

    ax_c2 = ax_c.twinx()
    l_ds = ax_c2.plot(
        g2_vals,
        ds_norm,
        "D--",
        color=c_ds_col,
        lw=1.8,
        ms=5.2,
        label=(
            r"Tórusz-normált $d_S^{\mathrm{norm}} \to 4$"
            if is_hu
            else r"Torus-normalized $d_S^{\mathrm{norm}} \to 4$"
        ),
    )
    ax_c2.set_ylim(3.65, 4.10)
    if is_hu:
        ax_c.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}".replace(".", ",")))
        ax_c2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}".replace(".", ",")))

    ax_c.set_xlabel(
        r"Csatolás $g_{\mathrm{grav}}^2 \to 0$ ($V=128$, $1\,536$ él, $800$ minta)"
        if is_hu
        else r"Coupling $g_{\mathrm{grav}}^2 \to 0$ ($V=128$, $1{,}536$ links, $800$ samples)"
    )
    ax_c.set_ylabel(r"$c_{\mathrm{DW}}^{\mathrm{eff}}$", color=c_dw_col)
    ax_c2.set_ylabel(r"$d_S^{\mathrm{norm}}$", color=c_ds_col)
    ax_c.set_title(
        r"(c) I modul: 4D $D_4$ él-MCMC ($c_{\mathrm{DW}}^{\mathrm{eff}} < 0$) és $d_S$"
        if is_hu
        else r"(c) Module I: 4D $D_4$ Link MCMC ($c_{\mathrm{DW}}^{\mathrm{eff}} < 0$) & $d_S$",
        fontweight="bold",
    )

    h1, lb1 = ax_c.get_legend_handles_labels()
    h2, lb2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, lb1 + lb2, loc="lower left", framealpha=0.92)

    note_c = (
        "Hawking--Luttrell skalaron $\\chi$: pozitív mag\n"
        "$+R_E^2$ előjelváltás: tachion NO-GO ($k_E^2=+M_{\\mathrm{inf}}^2$)\n"
        "$L_b=20$: $d_S^{\\mathrm{IR}}=4{,}041$, $d_S^{\\mathrm{UV}}=2{,}073$"
        if is_hu
        else "Hawking--Luttrell scalaron $\\chi$: positive kernel\n"
        "$+R_E^2$ sign flip: tachyon NO-GO ($k_E^2=+M_{\\mathrm{inf}}^2$)\n"
        "$L_b=20$: $d_S^{\\mathrm{IR}}=4.041$, $d_S^{\\mathrm{UV}}=2.073$"
    )
    ax_c.text(
        0.03,
        0.96,
        note_c,
        transform=ax_c.transAxes,
        va="top",
        ha="left",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    # ------------------------------------------------------------------
    # Panel (d): Module J — Wetterich FRG Shield & 10-Param Euclid TATT+AGN Ladder
    # ------------------------------------------------------------------
    f1000 = mod_j["gate1b_2c_euclid_10param_tatt_and_agn"]["euclid_10param_TATT_plus_AGN_ell30_1000_EuclidHighIA"]
    f1500 = mod_j["gate1b_2c_euclid_10param_tatt_and_agn"]["euclid_10param_TATT_plus_AGN_ell30_1500_EuclidHighIA"]

    stages = (
        [
            "1p\nrögzített",
            "7p\n+NLA",
            "8p\n+AGN",
            "9p\n+TATT",
            "10p TATT+AGN\n(Euclid)",
            "10p TATT+AGN\n(+Planck18)",
        ]
        if is_hu
        else [
            "1p\nfixed",
            "7p\n+NLA",
            "8p\n+AGN",
            "9p\n+TATT",
            "10p TATT+AGN\n(Euclid)",
            "10p TATT+AGN\n(+Planck18)",
        ]
    )
    vals_1000 = [
        f1000["SNR_1param_fixed"],
        f1000["SNR_7param_cosmo_plus_NLA"],
        f1000["SNR_8param_cosmo_plus_NLA_plus_AGN"],
        f1000["SNR_9param_cosmo_plus_TATT"],
        f1000["SNR_10param_Euclid_alone_TATT_plus_AGN"],
        f1000["SNR_10param_Euclid_plus_Planck18_TATT_plus_AGN"],
    ]
    vals_1500 = [
        f1500["SNR_1param_fixed"],
        f1500["SNR_7param_cosmo_plus_NLA"],
        f1500["SNR_8param_cosmo_plus_NLA_plus_AGN"],
        f1500["SNR_9param_cosmo_plus_TATT"],
        f1500["SNR_10param_Euclid_alone_TATT_plus_AGN"],
        f1500["SNR_10param_Euclid_plus_Planck18_TATT_plus_AGN"],
    ]

    x = np.arange(len(stages), dtype=float)
    w = 0.36
    b1 = ax_d.bar(
        x - 0.5 * w,
        vals_1000,
        width=w,
        color="#2874a6",
        edgecolor="#1b4f72",
        label=r"$\ell \in [30,\,1000]$ (High-IA)",
    )
    b2 = ax_d.bar(
        x + 0.5 * w,
        vals_1500,
        width=w,
        color="#5dade2",
        edgecolor="#1b4f72",
        label=r"$\ell \in [30,\,1500]$ (High-IA)",
    )
    ax_d.axhline(5.0, color="#b03a2e", ls="--", lw=1.3, label=r"$5\sigma$ / $3\sigma$ küszöb" if is_hu else r"$5\sigma$ / $3\sigma$ thresholds")
    ax_d.axhline(3.0, color="#ca6f1e", ls=":", lw=1.3)

    for rect in list(b1) + list(b2):
        h_v = rect.get_height()
        txt = hu_fmt(h_v, 2) if is_hu else f"{h_v:.2f}"
        ax_d.text(
            rect.get_x() + 0.5 * rect.get_width(),
            h_v + 0.14,
            txt,
            ha="center",
            va="bottom",
            fontsize=7.1,
        )

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(stages, fontsize=7.9)
    ax_d.set_ylim(0.0, 12.6)
    ax_d.grid(True, axis="y", ls=":", alpha=0.45)
    ax_d.set_ylabel(r"Diszformális $\alpha_{\mathrm{disf}}$ SNR [$\sigma$]" if is_hu else r"Disformal $\alpha_{\mathrm{disf}}$ SNR [$\sigma$]")
    ax_d.set_title(
        r"(d) J modul: Wetterich-pajzs és 10p Euclid TATT+AGN SNR"
        if is_hu
        else r"(d) Module J: Wetterich Shield & 10p Euclid TATT+AGN SNR",
        fontweight="bold",
    )
    ax_d.legend(loc="upper right", framealpha=0.94, fontsize=7.6)

    note_d = (
        "Korlátos $W_\\sigma$: $|\\Delta m_\\sigma^2|/m_\\sigma^2 = 8{,}04\\times 10^{-7}$\n"
        "1-csat. $u^4/(16\\pi^2)$: NO-GO ($12{,}36\\times$)\n"
        "12-él ansatz $d_c=\\frac{3}{4\\pi^2}u^4$: PASS"
        if is_hu
        else "Bounded $W_\\sigma$: $|\\Delta m_\\sigma^2|/m_\\sigma^2 = 8.04\\times 10^{-7}$\n"
        "1-ch $u^4/(16\\pi^2)$: NO-GO ($12.36\\times$)\n"
        "12-link ansatz $d_c=\\frac{3}{4\\pi^2}u^4$: PASS"
    )
    ax_d.text(
        0.02,
        0.97,
        note_d,
        transform=ax_d.transAxes,
        va="top",
        ha="left",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#cccccc", alpha=0.94),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def main() -> None:
    mod_h = json.loads((DATA_DIR / "spin10_dynamical_link_mc_and_continuum_scaling.json").read_text(encoding="utf-8"))
    mod_i = json.loads((DATA_DIR / "24cell_regge_lorentzian_continuum_limit_mc.json").read_text(encoding="utf-8"))
    mod_j = json.loads((DATA_DIR / "wetterich_frg_helmholtz_prefactors_and_tatt.json").read_text(encoding="utf-8"))

    en_pdf = FIG_DIR / "fig_continuum_limit_a_to_0_en.pdf"
    hu_pdf = FIG_DIR / "fig_continuum_limit_a_to_0_hu.pdf"
    build_figure("en", mod_h, mod_i, mod_j, en_pdf)
    build_figure("hu", mod_h, mod_i, mod_j, hu_pdf)
    print(f"Generated: {en_pdf.name}, {hu_pdf.name}")


if __name__ == "__main__":
    main()
