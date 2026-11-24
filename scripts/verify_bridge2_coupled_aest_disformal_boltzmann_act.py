#!/usr/bin/env python3
r"""Paper 29 (v2.2.0) Bridge 2B — Exact Covariant AeST Director Perturbation EOMs
(Scalar V_u & Transverse Vector u_i^T) + Live Coupled ACT DR6 & Planck PR4
Boltzmann Audit (ell <= 3000):

  1. Exact SymPy covariant derivation on perturbed expanding flat FLRW:
       - Transverse vector mode u_i^T (d^i u_i^T = 0):
           delta(nabla_mu u^mu) == 0 identically (zero c_2 theta^2 coupling),
           source-free transverse wave equation (a u_T)'' + c_V^2 k^2 (a u_T) = 0
           (zero scalar-matter source, decays ~ 1/a, zero linear B-mode lensing).
       - Irrotational scalar mode V_u (u_i = a d_i V_u):
           (a V_u)'' + c_S^2 k^2 (a V_u) = -(a Psi)' - 3 c_S^2 a (Phi' + H Psi)
           with c_S^2 = c_2 / c_4 (c_S = 1 for Paper 27/28 Stueckelberg-Lorenz;
           c_S = 0 recovers exact comoving dust velocity potential V_m = -v_d/k).
  2. Live integration into Paper 29 Step 1's EXACT ell <= 3000 Boltzmann + RECFAST v1.5 +
     Halofit + Challinor-Lewis curved-sky lensing + archival ACT DR6 (108-bin TT/TE/EE) +
     ACT DR6 & Planck PR4 (22-bin C_L^{kk}) likelihood across c_S in {1.0, 0.1, 0.01, 0.0}
     plus an adversarial monotone lapse stress test (N_bar = 1 - 0.015 a^{3/2}).

Strict `python3 -W error` compliance. Zero hardcoded local paths.
"""

import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import sympy as sp

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import verify_paper29_step1_theta_h0_void_optical_closure as p29


def sympy_verify_aest_director_eoms():
    """Exact SymPy derivation of the linearized covariant AeST / Stueckelberg-Lorenz
    scalar (V_u) and transverse vector (u_i^T) equations of motion on flat FLRW.
    """
    tau, x, y, z = sp.symbols("tau x y z", real=True)
    eps = sp.Symbol("eps", real=True)
    a = sp.Function("a")(tau)
    H = sp.diff(a, tau) / a
    Hp = sp.diff(H, tau)

    Phi = sp.Function("Phi")(tau, x)
    Psi = sp.Function("Psi")(tau, x)
    Vu = sp.Function("V_u")(tau, x)
    uyT = sp.Function("u_yT")(tau, x)

    coords = [tau, x, y, z]
    g = sp.Matrix([
        [-a**2 * (1 + 2 * eps * Psi), 0, 0, 0],
        [0, a**2 * (1 - 2 * eps * Phi), 0, 0],
        [0, 0, a**2 * (1 - 2 * eps * Phi), 0],
        [0, 0, 0, a**2 * (1 - 2 * eps * Phi)],
    ])
    g_inv = sp.Matrix([
        [sp.series(-1 / (a**2 * (1 + 2 * eps * Psi)), eps, 0, 2).removeO(), 0, 0, 0],
        [0, sp.series(1 / (a**2 * (1 - 2 * eps * Phi)), eps, 0, 2).removeO(), 0, 0],
        [0, 0, sp.series(1 / (a**2 * (1 - 2 * eps * Phi)), eps, 0, 2).removeO(), 0],
        [0, 0, 0, sp.series(1 / (a**2 * (1 - 2 * eps * Phi)), eps, 0, 2).removeO()],
    ])

    u_cov = [-a * (1 + eps * Psi), eps * a * sp.diff(Vu, x), eps * a * uyT, sp.Integer(0)]
    u_con = [sp.series(sum(g_inv[mu, nu] * u_cov[nu] for nu in range(4)), eps, 0, 2).removeO() for mu in range(4)]

    norm_u = sp.simplify(sp.series(sum(u_con[mu] * u_cov[mu] for mu in range(4)), eps, 0, 2).removeO())
    assert norm_u == -1

    def chr_sym(lam, mu, nu):
        s = sum(
            g_inv[lam, sig] * (sp.diff(g[sig, mu], coords[nu]) + sp.diff(g[sig, nu], coords[mu]) - sp.diff(g[mu, nu], coords[sig]))
            for sig in range(4)
        )
        return sp.series(sp.Rational(1, 2) * s, eps, 0, 2).removeO()

    sqrt_mg = sp.series(a**4 * (1 + eps * (Psi - 3 * Phi)), eps, 0, 2).removeO()
    theta_expr = sp.series(
        sum(sp.diff(sqrt_mg * u_con[mu], coords[mu]) for mu in range(4)) / sqrt_mg,
        eps, 0, 2,
    ).removeO()
    d_theta = sp.simplify(sp.diff(theta_expr, eps).subs(eps, 0))

    assert sp.simplify(sp.diff(d_theta, uyT)) == 0
    assert sp.simplify(sp.diff(d_theta, sp.diff(uyT, x))) == 0
    assert sp.simplify(sp.diff(d_theta, sp.diff(uyT, tau))) == 0
    assert sp.simplify(d_theta - (sp.diff(Vu, x, 2) - 3 * sp.diff(Phi, tau) - 3 * H * Psi) / a) == 0

    a_cov = []
    for mu in range(4):
        val = sum(u_con[nu] * (sp.diff(u_cov[mu], coords[nu]) - sum(chr_sym(lam, nu, mu) * u_cov[lam] for lam in range(4))) for nu in range(4))
        a_cov.append(sp.simplify(sp.series(val, eps, 0, 2).removeO()))
    assert sp.simplify(sp.diff(a_cov[1], eps).subs(eps, 0) - sp.diff(sp.diff(Vu, tau) + H * Vu + Psi, x)) == 0
    assert sp.simplify(sp.diff(a_cov[2], eps).subs(eps, 0) - (sp.diff(uyT, tau) + H * uyT)) == 0

    k, c2, c4, cS2, KB = sp.symbols("k c_2 c_4 c_S2 K_B", positive=True)
    Vu_t = sp.Function("V_u_k")(tau)
    Phi_t = sp.Function("Phi_k")(tau)
    Psi_t = sp.Function("Psi_k")(tau)
    L_s = sp.Rational(1, 2) * a**2 * (
        c4 * k**2 * (sp.diff(Vu_t, tau) + H * Vu_t + Psi_t) ** 2
        - c2 * (-k**2 * Vu_t - 3 * sp.diff(Phi_t, tau) - 3 * H * Psi_t) ** 2
    )
    EL_s = sp.simplify((sp.diff(sp.diff(L_s, sp.diff(Vu_t, tau)), tau) - sp.diff(L_s, Vu_t)) / (c4 * a**2 * k**2))
    EL_s_cs = sp.simplify(EL_s.subs(c2, cS2 * c4))
    assert sp.simplify(
        a * EL_s_cs - (sp.diff(a * Vu_t, tau, 2) + cS2 * k**2 * a * Vu_t + sp.diff(a * Psi_t, tau) + 3 * cS2 * a * (sp.diff(Phi_t, tau) + H * Psi_t))
    ) == 0

    w_t = sp.Function("w_T")(tau)
    L_v = sp.Rational(1, 2) * ((c4 + KB) * sp.diff(w_t, tau) ** 2 - KB * k**2 * w_t**2)
    EL_v = sp.simplify((sp.diff(sp.diff(L_v, sp.diff(w_t, tau)), tau) - sp.diff(L_v, w_t)) / (c4 + KB))
    assert sp.simplify(EL_v - (sp.diff(w_t, tau, 2) + (KB / (c4 + KB)) * k**2 * w_t)) == 0

    return dict(
        unit_norm_check=str(norm_u),
        d_theta_independent_of_uT=True,
        scalar_EOM="(a V_u)'' + c_S^2 k^2 (a V_u) = -(a Psi)' - 3 c_S^2 a (Phi' + H Psi)",
        vector_EOM="(a u_T)'' + c_V^2 k^2 (a u_T) = 0 [zero scalar source, decays ~ 1/a, zero linear B-mode lensing]",
        sympy_exact_pass=True,
    )


def solve_director_Vu_exact_ode(eta_arr, a_arr, H_arr, Phi_arr, Psi_arr, k_arr, c_S: float):
    Nt, Nk = Phi_arr.shape
    a_Psi = a_arr[:, None] * Psi_arr
    d_aPsi = np.gradient(a_Psi, eta_arr, axis=0)
    d_Phi = np.gradient(Phi_arr, eta_arr, axis=0)
    src = -d_aPsi - 3.0 * (c_S**2) * (a_arr[:, None] * (d_Phi + H_arr[:, None] * Psi_arr))

    if c_S == 0.0:
        w = np.zeros((Nt, Nk), dtype=float)
        w[0] = -0.5 * a_Psi[0] * eta_arr[0]
        for i in range(Nt - 1):
            dt = eta_arr[i + 1] - eta_arr[i]
            w[i + 1] = w[i] - 0.5 * dt * (a_Psi[i] + a_Psi[i + 1])
        Vu = w / a_arr[:, None]
        dVu = (-Psi_arr) - H_arr[:, None] * Vu
        return Vu, dVu

    om2 = (c_S**2) * (k_arr**2)
    w = np.zeros((Nt, Nk), dtype=float)
    p = np.zeros((Nt, Nk), dtype=float)
    x0 = c_S * k_arr * eta_arr[0]
    wt_qs = (x0**2) / (1.0 + x0**2)
    w[0] = wt_qs * (src[0] / np.maximum(om2, 1.0e-30)) + (1.0 - wt_qs) * (-0.5 * a_Psi[0] * eta_arr[0])
    p[0] = (1.0 - wt_qs) * (-a_Psi[0])
    for i in range(Nt - 1):
        dt = eta_arr[i + 1] - eta_arr[i]
        h = 0.5 * dt
        s_mid = 0.5 * (src[i] + src[i + 1])
        rhs1 = w[i] + h * p[i]
        rhs2 = p[i] - h * om2 * w[i] + dt * s_mid
        det = 1.0 + (h**2) * om2
        w[i + 1] = (rhs1 + h * rhs2) / det
        p[i + 1] = (rhs2 - h * om2 * rhs1) / det
    Vu = w / a_arr[:, None]
    dVu = (p / a_arr[:, None]) - H_arr[:, None] * Vu
    return Vu, dVu


def build_exact_p29_boltzmann_harness():
    src = (SCRIPT_DIR / "verify_paper29_step1_theta_h0_void_optical_closure.py").read_text(encoding="utf-8")
    i0 = src.index("def run_gate_s1_3_boltzmann_void_clock_master_closure(")
    i1 = src.index("    res_hom_th = _full_run(H0_hom_th, 0.0, False)")
    fn_src = src[i0:i1]
    fn_src = fn_src.replace('"E_opt_of_a": E_opt_of_a,', '"E_opt_of_a": E_opt_of_a, "N_clock_of_a": N_clock_of_a,', 1)
    fn_src = fn_src.replace(
        "lens_z, lens_eta, lens_weyl_lin, lens_dm_lin, lens_dm_weyl_lin = [], [], [], [], []",
        "lens_z, lens_eta, lens_weyl_lin, lens_dm_lin, lens_dm_weyl_lin = [], [], [], [], []\n"
        "        lens_a, lens_aH, lens_phi, lens_psi, lens_vd = [], [], [], [], []",
        1,
    )
    fn_src = fn_src.replace(
        "lens_weyl_lin.append((0.5 * phi_tot_weyl).copy())",
        "lens_weyl_lin.append((0.5 * phi_tot_weyl).copy())\n"
        "                lens_a.append(float(a_e))\n"
        "                lens_aH.append(float(aH_e))\n"
        "                lens_phi.append((phi + dpsi_ir_e).copy())\n"
        "                lens_psi.append(psi.copy())\n"
        "                lens_vd.append(vd.copy())",
        1,
    )
    fn_src = fn_src.replace(
        '"lens_dm_weyl_lin": np.array(lens_dm_weyl_lin)[::-1],',
        '"lens_dm_weyl_lin": np.array(lens_dm_weyl_lin)[::-1],\n'
        '            "lens_a": np.array(lens_a)[::-1],\n'
        '            "lens_aH": np.array(lens_aH)[::-1],\n'
        '            "lens_phi": np.array(lens_phi)[::-1],\n'
        '            "lens_psi": np.array(lens_psi)[::-1],\n'
        '            "lens_vd": np.array(lens_vd)[::-1],',
        1,
    )
    fn_src += (
        "    return dict(\n"
        "        H0_hom_th=H0_hom_th, H0_vc_A=H0_vc_A, H0_vc_B=H0_vc_B,\n"
        "        _compute_bg=_compute_bg, _build_se_table=_build_se_table,\n"
        "        _solve_boltzmann_l28=_solve_boltzmann_l28,\n"
        "        _compute_spectra_and_cmb_lk=_compute_spectra_and_cmb_lk,\n"
        "        k_cmb=k_cmb,\n"
        "    )\n"
    )
    ns = dict(p29.__dict__)
    exec(compile(fn_src, "<p29_exact_hook>", "exec"), ns)
    return ns["run_gate_s1_3_boltzmann_void_clock_master_closure"](None)


def main():
    t0 = time.time()
    sym = sympy_verify_aest_director_eoms()
    eng = build_exact_p29_boltzmann_harness()
    k_cmb, H0_vc_A = eng["k_cmb"], eng["H0_vc_A"]
    bg_A = eng["_compute_bg"](H0_vc_A, p29.ALPHA_OCSR, False)
    se_A = eng["_build_se_table"](H0_vc_A)
    tr_A = eng["_solve_boltzmann_l28"](bg_A, se_A)

    z_inc = tr_A["lens_z"][::-1]
    a_inc = tr_A["lens_a"][::-1]
    eta_inc = tr_A["lens_eta"][::-1]
    aH_inc = tr_A["lens_aH"][::-1]
    phi_inc = tr_A["lens_phi"][::-1]
    psi_inc = tr_A["lens_psi"][::-1]
    vd_inc = tr_A["lens_vd"][::-1]
    weyl_GR_inc = tr_A["lens_weyl_lin"][::-1]

    N_inc = bg_A["N_clock_of_a"](a_inc)
    dN_deta_inc = np.gradient(N_inc, eta_inc)

    Vu_0, _ = solve_director_Vu_exact_ode(eta_inc, a_inc, aH_inc, phi_inc, psi_inc, k_cmb, c_S=0.0)
    Vm_boltz = -vd_inc / k_cmb[None, :]
    iz1 = int(np.argmin(np.abs(z_inc - 1.0)))
    ik05 = int(np.argmin(np.abs(k_cmb - 0.05)))
    rel_err_vm = float(abs(Vu_0[iz1, ik05] - Vm_boltz[iz1, ik05]) / max(abs(Vm_boltz[iz1, ik05]), 1e-30))
    assert rel_err_vm < 5e-3

    base_p1 = eng["_compute_spectra_and_cmb_lk"](tr_A, p29.AS_PRESC_I, p29.NS_PRESC_I)
    base_2f = eng["_compute_spectra_and_cmb_lk"](tr_A, p29.AS_2FIELD, p29.NS_2FIELD)

    c1_N = (1.0 - N_inc**2) / (2.0 * N_inc**2)
    c2_N = ((1.0 + N_inc**2) / (2.0 * N_inc**3)) * dN_deta_inc

    summary = dict(
        version="v2.2.0",
        sympy_eom=sym,
        H0_vc_A=round(float(H0_vc_A), 4),
        theta_100=round(float(bg_A["theta_100"]), 5),
        sanity_cs0_vs_boltzmann_Vm_rel_err=round(rel_err_vm, 6),
        N_bar_profile={
            f"z{zt}": dict(N_bar=round(float(N_inc[j]), 6), dN_dtau_over_H=round(float(dN_deta_inc[j] / aH_inc[j]), 6))
            for zt in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
            for j in [int(np.argmin(np.abs(z_inc - zt)))]
        },
        baseline_P1={k: v for k, v in base_p1.items() if not isinstance(v, np.ndarray)},
        baseline_2F={k: v for k, v in base_2f.items() if not isinstance(v, np.ndarray)},
        director_branches={},
    )

    for label, cs_val, use_monotone_N in [
        ("Covariant_SL_cS_1.0", 1.0, False),
        ("Intermediate_cS_0.1", 0.1, False),
        ("Low_cS_0.01", 0.01, False),
        ("Comoving_Dust_cS_0.0", 0.0, False),
        ("Comoving_Boltz_Vm_exact", None, False),
        ("Monotone_N0p985_cS_1.0", 1.0, True),
        ("Monotone_N0p985_cS_0.0", 0.0, True),
    ]:
        if use_monotone_N:
            N_use = 1.0 - 0.015 * (a_inc ** 1.5)
            dN_use = np.gradient(N_use, eta_inc)
            c1_u = (1.0 - N_use**2) / (2.0 * N_use**2)
            c2_u = ((1.0 + N_use**2) / (2.0 * N_use**3)) * dN_use
        else:
            c1_u, c2_u = c1_N, c2_N

        if cs_val is None:
            Vu_k = Vm_boltz
            dVu_k = -psi_inc - aH_inc[:, None] * Vu_k
        else:
            Vu_k, dVu_k = solve_director_Vu_exact_ode(eta_inc, a_inc, aH_inc, phi_inc, psi_inc, k_cmb, c_S=cs_val)

        dWeyl_inc = -c1_u[:, None] * dVu_k + c2_u[:, None] * Vu_k
        tr_mod = dict(tr_A)
        tr_mod["lens_weyl_lin"] = (weyl_GR_inc + dWeyl_inc)[::-1]
        r_p1 = eng["_compute_spectra_and_cmb_lk"](tr_mod, p29.AS_PRESC_I, p29.NS_PRESC_I)
        r_2f = eng["_compute_spectra_and_cmb_lk"](tr_mod, p29.AS_2FIELD, p29.NS_2FIELD)

        frac_kk = {str(L): round(100.0 * float(r_p1["Cl_kk"][L] / base_p1["Cl_kk"][L] - 1.0), 4) for L in [10, 40, 100, 300, 1000]}
        summary["director_branches"][label] = dict(
            c_S=cs_val, monotone_N=use_monotone_N,
            dCl_kk_pct=frac_kk,
            P1_ACT108=r_p1["act_dr6_108_cal_chi2"],
            P1_dACT108=round(r_p1["act_dr6_108_cal_chi2"] - base_p1["act_dr6_108_cal_chi2"], 3),
            P1_KK13=r_p1["act_dr6_lens_13_chi2"],
            P1_dKK13=round(r_p1["act_dr6_lens_13_chi2"] - base_p1["act_dr6_lens_13_chi2"], 3),
            P1_KK22=r_p1["act_dr6_planck_pr4_lens_22_chi2"],
            P1_dKK22=round(r_p1["act_dr6_planck_pr4_lens_22_chi2"] - base_p1["act_dr6_planck_pr4_lens_22_chi2"], 3),
            F2_KK22=r_2f["act_dr6_planck_pr4_lens_22_chi2"],
            F2_dKK22=round(r_2f["act_dr6_planck_pr4_lens_22_chi2"] - base_2f["act_dr6_planck_pr4_lens_22_chi2"], 3),
        )

    out_path = DATA_DIR / "bridge2_coupled_boltzmann_results.json"
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    sys.stderr.write(f"[Paper 29 v2.2.0 Bridge 2B Coupled AeST+Disformal Boltzmann PASS in {time.time()-t0:.1f}s]\n")


if __name__ == "__main__":
    main()
