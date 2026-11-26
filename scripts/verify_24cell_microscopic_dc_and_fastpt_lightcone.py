#!/usr/bin/env python3
"""
Paper 29 (Addendum v3.6 — Module K, Rev 2): Microscopic 4D 24-Cell (D_4) Lattice Audit of (d_max, d_c),
Exact 1-Loop FAST-PT (Blazek et al. 2019) 11-Parameter TATT Forecast, and Coles-Jones Lightcone Mocks
========================================================================================================

Addresses the UV-to-IR Helmholtz-screening and Stage-IV weak-lensing items localized in Module J (v3.5.0):

  1. Microscopic 4D 24-Cell (D_4) 12-Link Lattice Action & Scale-Matching Audit for (d_max, d_c):
     - Reproduces two external 4D lattice-QFT literature benchmarks:
       (a) Luscher & Weisz (1995) / Capitani (2003, Phys. Rept. 382, 113, Table 1) 17-digit 4D hypercubic
           Z^4 massless tadpole (Watson) integral Z_0(Z^4) = 0.15493339023106021 (|err| < 10^{-15});
       (b) Hasenfratz & Karsch (1983, Phys. Lett. B 125, 308) 4D lattice chemical-potential theorem:
           naive linear coupling generates a non-zero O(a^{-2}) vacuum susceptibility divergence, whereas
           exact link-exponential hopping (1/2)(e^{+theta} T_{+e_m} + e^{-theta} T_{-e_m}) cancels the
           vacuum O(a^{-2}) UV divergence identically (< 10^{-14}) on BOTH Z^4 and the 24-cell (D_4) lattice.
     - Derives in SymPy from the discrete K_4 subset 24-cell traceless defect operator T_def = I_4 - H_b
       (spec = {+5 u, -3 u, -u, -u}, u = sqrt(3)/24):
       * the exact transverse SO(2) spatial director quadratic Casimir d_max = (1/2) Tr(Pi_perp T_def^2)
         = u^2 = 1/192 = 5.2083333e-3;
       * the exact 12-link seed identity N_links * u^2 = 12 * (1/192) = 1/16 == r_seed^2;
       * Hasenfratz-Karsch link-reversal parity e_m <-> -e_m across the 12 antipodal pairs of the 24-cell
         generating K_NS(sigma_scr, Q) = 2 K_2 Q_0^2 d_eff^2 [cosh(Delta_Q / d_eff) - 1];
       * the exact Voronoi cell-volume Jacobian V_0 = 1/2 mapping dimensionless lattice variance Z_0(D_4)
         to the canonical continuum scalar tadpole G_{D_4}(0; a) = Z_0(D_4) / (V_0 a^2) = 2 Z_0(D_4) / a^2.
     - Computes the exact 4D 24-cell (D_4) lattice Green's function (Watson integral)
           Z_0(D_4) = <1 / L_D4>_{BZ} = 0.138230474737
       via TWO independent methods (Richardson O(m^{-2}, m^{-4}) Brillouin-zone extrapolation across
       m in {24, 32, 48, 64, 96} AND exact 1D Hubbard-Stratonovich + Schwinger proper-time integration,
       agreeing to < 10^{-9}), plus proves the W(F_4)-invariant per-link resolvent identity
       Pi_{mm}^{(D_4)} = < |Delta_m chi|^2 / L_D4^2 >_{BZ} == (1/4) Z_0(D_4) across all 12 links.
     - Evaluates the explicit quadratic link-strain modulus hat{D}(x) across bare vs matched UV scales:
       (i)   PROVEN NO-GO at naive bare edge a = l_* = M_*^{-1} (a M_* = 1): BOTH the 12-link sum
             (d_c = 24 Z_0(D_4) u^4 = 8.9994e-5, 37.34x Cosmic-Jeans violation) AND even the single
             W(F_4)-invariant radial-breathing singlet (d_c = 2 Z_0(D_4) u^4 = 7.4995e-6, 3.11x violation;
             and 2.4998e-6 > 2.41e-6 at a M_* = sqrt(3)) violate the Bridge-2D Cosmic-Jeans upper bound!
       (ii)  CONDITIONAL Candidate A (12-link heat-kernel tadpole matching G_{D_4}(0; a) == M_*^2 / (16 pi^2),
             fixing a M_* = 4 pi sqrt(2 Z_0(D_4)) = 6.607342, Debye Lambda_{BZ} = 0.7587 M_*):
             Z_0(D_4) cancels algebraically to yield d_c = (3 / (4 pi^2)) u^4 = (m_R / M_*) d_max = 2.061385e-6.
       (iii) CONDITIONAL Candidate B (1-mode radial-breathing singlet at 24-cell circumdiameter scale
             a = 2 l_* = 2 M_*^{-1}, i.e. a M_* = 2 inside the viable window a M_* in [1.764, 2.236]):
             d_c = [2 Z_0(D_4) / (a M_*)^2] u^4|_{a M_* = 2} = (1/2) Z_0(D_4) u^4 = 1.874871e-6,
             verified to < 10^{-12} Schur residual across all 4,001 radial shells in Module F's 3-field BVP!

  2. Complete 1-Loop Perturbation Theory (FAST-PT / Blazek et al. 2019, PRD 100, 103506) Mode-Coupling &
     Propagator Integrals + Exact Nonlocal 11-Parameter (b_TA-Included) Euclid Stage-IV Fisher Forecast:
     - Reproduces Makino-Sasaki-Suto (1992, PRD 46, 585) P_{22}(1) + P_{13}(1) == 55 / 784 = 0.07015306
       from Goroff-Grinstein-Rey-Wise SymPy recursion, proving exact Galilean 1/r^2 IR pole cancellation.
     - Derives ALL Blazek et al. (2019, Appendix A) TATT 1-loop mode-coupling AND 1PR propagator kernels
       in SymPy — INCLUDING BOTH P_{0|0E} = 2 A_{0|0E} + B_{0|0E}^{ren} AND the complete quadratic-TT
       cross-spectrum P_{0|E2} = 2 A_{0|E2} + 4 B_{0|E2}^{ren} (proving its exact closed-form rational+log
       formula, r -> inf limit +58/315, and Galilean IR cancellation 32/315) — and re-evaluates the full
       nonlocal 1-loop convolutions inside every cosmological Fisher variation (Omega_{m,0}, sigma_8, n_s, h)!

  3. 64-Realization 24-Shell Coles-Jones (1991) Log-Normal + Resolution-Independent FFT-TATT Lightcones:
     - Uses the exact Coles & Jones (1991, MNRAS 248, 1) inverse log-normal correlation transform
       xi_G(r_perp) = ln(1 + xi_target(r_perp)) on 160x160 patches (12 deg x 12 deg, ell_fund = 30.0,
       inscribed ell_Nyq = 1,200 > 1,000), smooths shell tidal fields at k_sm = 1.0 h Mpc^{-1} with exact
       resolution-independent 3D-to-2D Lagrangian variance matching, calibrates noiseless templates and
       Hartlap-debiased independent GR baseline covariances (zero LOO tautology!), and measures clean
       non-Gaussian + nonlinear-TATT variance inflation against a paired 64-mock Gaussian shell control.

Strictly follows the 4-Gate Physics Calculation Protocol (AGENTS.md) with zero machine-specific paths.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import numpy as np
import sympy as sp
from scipy import integrate, interpolate, special

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_P29_REPO = (
    SCRIPT_DIR.parent
    if (SCRIPT_DIR / "verify_paper29_step1_theta_h0_void_optical_closure.py").exists()
    else SCRIPT_DIR.parent.parent / "tau-core-frontier-closures-paper29"
)
DATA_DIR = _P29_REPO / "data"
for _p in (SCRIPT_DIR, _P29_REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import verify_paper29_step1_theta_h0_void_optical_closure as p29
import verify_bridge2_coupled_aest_disformal_boltzmann_act as b2b
import verify_bridge2c_lowz_weak_lensing_des_kids_euclid as b2c
import verify_euclid_ia_cosmo_marginalization_and_isw as mod_c
import verify_uv_helmholtz_couplings_and_joint_2field_bvp as mod_f


def mm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Warning-free matrix multiplication under python3 -W error (avoids macOS Accelerate matmul flags)."""
    return np.einsum("ij,jk->ik", A, B, optimize=True)


def build_24cell_12_undirected_links() -> np.ndarray:
    """Construct the 12 undirected unit nearest-neighbor link vectors e_m in R^4 of the 24-cell (D_4)."""
    links = []
    for i in range(4):
        for j in range(i + 1, 4):
            for s2 in (-1.0, 1.0):
                v = np.zeros(4, dtype=np.float64)
                v[i] = 1.0 / math.sqrt(2.0)
                v[j] = s2 / math.sqrt(2.0)
                links.append(v)
    E12 = np.array(links, dtype=np.float64)
    assert E12.shape == (12, 4)
    I4 = np.eye(4, dtype=np.float64)
    T2 = np.einsum("ma,mb->ab", E12, E12)
    assert np.max(np.abs(T2 - 3.0 * I4)) < 1.0e-15
    T4 = np.einsum("ma,mb,mc,md->abcd", E12, E12, E12, E12)
    T4_iso = 0.5 * (
        np.einsum("ab,cd->abcd", I4, I4)
        + np.einsum("ac,bd->abcd", I4, I4)
        + np.einsum("ad,bc->abcd", I4, I4)
    )
    assert np.max(np.abs(T4 - T4_iso)) < 1.0e-15
    return E12


def gate1a_and_2a_24cell_microscopic_dc_derivation() -> dict:
    """Gates 1A, 2A & 3A:
      1. Gate 1A External 4D Lattice QFT Benchmarks:
         - Luscher & Weisz (1995, NPB 452, 213) / Capitani (2003, Phys. Rept. 382, 113, Table 1):
           4D Z^4 massless tadpole Z_0(Z^4) == 0.15493339023106021 (verified to < 10^{-15}).
         - Hasenfratz & Karsch (1983, Phys. Lett. B 125, 308): exact cancellation of O(a^{-2})
           vacuum susceptibility divergence for link-exponential hopping exp(+-theta) vs. non-zero
           UV divergence for naive linear theta coupling on BOTH Z^4 and 24-cell (D_4).
      2. Theorem K1 (SymPy): Exact K_4 subset 24-cell transverse SO(2) defect eigenvalue
         d_max = (1/2) Tr(Pi_perp T_def^2) = u^2 = 1/192, 12-link seed identity 12 u^2 == r_seed^2 = 1/16,
         Hasenfratz-Karsch link-reversal parity origin of 2 K_2 Q_0^2 d_eff^2 [cosh(Delta_Q / d_eff) - 1],
         and exact Voronoi cell-volume Jacobian V_0 = 1/2 => G_{D_4}(0; a) = Z_0(D_4) / (V_0 a^2) = 2 Z_0(D_4) / a^2.
      3. Theorem K2 (Exact 4D D_4 Watson Integral & Bare vs. Matched UV Scale Audit):
         - Cross-verifies Z_0(D_4) = 0.138230474737 via Richardson O(m^{-2}, m^{-4}) BZ grids AND
           exact 1D Hubbard-Stratonovich Schwinger proper-time integration (< 10^{-9} agreement!).
         - Proves W(F_4) link-resolvent identity Pi_{mm}^{(D_4)} == (1/4) Z_0(D_4) for all m = 1..12.
         - Proves PROVEN NO-GO at naive bare edge a = l_* = M_*^{-1} (a M_* = 1) for BOTH 12-link
           (8.9994e-5, 37.34x Cosmic-Jeans violation) AND 1-mode radial singlet (7.4995e-6, 3.11x violation),
           and evaluates both surviving CONDITIONAL scale matchings: Candidate A (12-link heat-kernel
           tadpole matching a M_* = 6.607342 -> d_c = (3/(4 pi^2)) u^4 = 2.061385e-6) and Candidate B
           (1-mode radial singlet at circumdiameter scale a = 2 l_* -> d_c = (1/2) Z_0(D_4) u^4 = 1.874871e-6).
    """
    # -------------------------------------------------------------------------
    # Gate 1A.1: Luscher-Weisz / Capitani (2003) 4D Z^4 Watson integral Z_0(Z^4)
    # -------------------------------------------------------------------------
    CAPITANI_2003_Z0_Z4 = 0.15493339023106021
    z0_z4_num, _ = integrate.quad(
        lambda t: special.i0e(2.0 * t) ** 4, 0.0, np.inf, epsabs=1.0e-14, epsrel=1.0e-14, limit=250
    )
    err_z0_z4 = abs(z0_z4_num - CAPITANI_2003_Z0_Z4)
    assert err_z0_z4 < 1.0e-15, f"Capitani Z_0(Z^4) benchmark failed: {z0_z4_num} vs {CAPITANI_2003_Z0_Z4}"

    # -------------------------------------------------------------------------
    # Gate 1A.2: Hasenfratz & Karsch (1983, PLB 125, 308) 4D Lattice UV Theorem
    # -------------------------------------------------------------------------
    E12 = build_24cell_12_undirected_links()
    m_hk = 32
    th_hk = (np.arange(m_hk, dtype=np.float64) + 0.5) * (2.0 * math.pi / m_hk) - math.pi

    q0_s, Lrest_s, mg2_s, cm_s = sp.symbols("q0 L_rest m_gap2 c_m", real=True)
    M_s = Lrest_s + 2 * (1 - sp.cos(cm_s * q0_s)) + mg2_s
    V1_s = -2 * sp.I * cm_s * sp.sin(cm_s * q0_s)
    V2_s = -2 * (cm_s ** 2) * sp.cos(cm_s * q0_s)
    tot_deriv_s = sp.diff(sp.diff(M_s, q0_s) / M_s, q0_s)
    assert sp.simplify(tot_deriv_s + (V2_s / M_s - (V1_s ** 2) / (M_s ** 2))) == 0

    m_gap_sq = 2.0
    Q0z, Q1z, Q2z, Q3z = np.meshgrid(th_hk, th_hk, th_hk, th_hk, indexing="ij")
    Mz4 = (
        2.0 * (1.0 - np.cos(Q0z))
        + 2.0 * (1.0 - np.cos(Q1z))
        + 2.0 * (1.0 - np.cos(Q2z))
        + 2.0 * (1.0 - np.cos(Q3z))
        + m_gap_sq
    ).ravel()
    V1_z4_sq = -4.0 * (np.sin(Q0z.ravel()) ** 2)
    V2_z4_hk = -2.0 * np.cos(Q0z.ravel())
    chi_hk_z4 = 0.5 * float(np.mean(V2_z4_hk / Mz4 - V1_z4_sq / (Mz4 ** 2)))
    chi_naive_z4 = 0.5 * float(np.mean(-V1_z4_sq / (Mz4 ** 2)))
    assert abs(chi_hk_z4) < 1.0e-14, f"Hasenfratz-Karsch Z^4 UV cancellation failed: {chi_hk_z4}"
    assert chi_naive_z4 > 0.01, f"Naive linear coupling on Z^4 failed to diverge: {chi_naive_z4}"

    Q_d4_hk = np.stack([Q0z.ravel(), Q1z.ravel(), Q2z.ravel(), Q3z.ravel()], axis=1) * math.sqrt(2.0)
    qe_hk = mm(Q_d4_hk, E12.T)
    Md4_hk = (2.0 / 3.0) * np.sum(1.0 - np.cos(qe_hk), axis=1) + m_gap_sq
    c_m0 = math.sqrt(2.0) * E12[:, 0]
    V1_d4 = -(2.0 / 3.0) * 1j * np.sum(c_m0[None, :] * np.sin(qe_hk), axis=1)
    V2_d4_hk = -(2.0 / 3.0) * np.sum((c_m0[None, :] ** 2) * np.cos(qe_hk), axis=1)
    chi_hk_d4 = 0.5 * float(np.mean((V2_d4_hk / Md4_hk - (V1_d4 ** 2) / (Md4_hk ** 2)).real))
    chi_naive_d4 = 0.5 * float(np.mean((-(V1_d4 ** 2) / (Md4_hk ** 2)).real))
    assert abs(chi_hk_d4) < 1.0e-14, f"Hasenfratz-Karsch D_4 UV cancellation failed: {chi_hk_d4}"
    assert chi_naive_d4 > 0.01

    # -------------------------------------------------------------------------
    # Gate 2A.1 (SymPy): K_4 Casimir, HK Potential & Exact Voronoi V_0 = 1/2 Jacobian
    # -------------------------------------------------------------------------
    Hb = sp.Matrix([
        [1, -sp.sqrt(5) / 8, 0, 0],
        [-sp.sqrt(5) / 8, 1 - sp.sqrt(3) / 12, 0, 0],
        [0, 0, 1 + sp.sqrt(3) / 24, 0],
        [0, 0, 0, 1 + sp.sqrt(3) / 24],
    ])
    T_def = sp.eye(4) - Hb
    u_sym = sp.sqrt(3) / 24
    Pi_perp = sp.diag(0, 0, 1, 1)
    dmax_sym = sp.simplify(sp.trace(Pi_perp * (T_def ** 2)) / sp.trace(Pi_perp))
    r_seed_sym = sp.simplify(2 * sp.sqrt(3) * u_sym)
    assert sp.simplify(dmax_sym - (u_sym ** 2)) == 0
    assert dmax_sym == sp.Rational(1, 192)
    assert r_seed_sym == sp.Rational(1, 4)
    assert sp.simplify(12 * dmax_sym - (r_seed_sym ** 2)) == 0

    DQ_s, Q0_s, K2_s, deff_s = sp.symbols("Delta_Q Q_0 K_2 d_eff", positive=True)
    hop_sym = sp.Rational(1, 2) * (sp.exp(DQ_s / deff_s) + sp.exp(-DQ_s / deff_s)) - 1
    assert sp.simplify(hop_sym - (sp.cosh(DQ_s / deff_s) - 1)) == 0
    V0_amp = sp.Symbol("V_0_amp", positive=True)
    K_cand = V0_amp * hop_sym
    quad_coeff = sp.simplify(sp.Rational(1, 2) * sp.diff(K_cand, DQ_s, 2).subs(DQ_s, 0) / (Q0_s ** 2))
    V0_sol = sp.solve(sp.Eq(quad_coeff, K2_s), V0_amp)[0]
    assert sp.simplify(V0_sol - 2 * K2_s * (Q0_s ** 2) * (deff_s ** 2)) == 0

    # First-principles SymPy derivation of the Voronoi 24-cell volume Jacobian V_0 = 1/2:
    # Standard integer D_4 = {n in Z^4 : sum n_mu in 2 Z} has index 2 in Z^4 (det = 2) and root norm sqrt(2).
    # Rescaling by 1/sqrt(2) so nearest-neighbor edge length is |e_m| = a gives Voronoi cell 4-volume
    # V_cell(a) = 2 * (a / sqrt(2))^4 = (1/2) a^4 = V_0 a^4 with V_0 = 1/2!
    # A dimensionless lattice scalar tilde_phi_x with unit-weight action
    #   S_lat = (1/2) sum_x tilde_phi_x (a^2 L_D4(a)) tilde_phi_x -> (1/2) int [d^4x / (V_0 a^4)] a^2 (d tilde_phi)^2
    # matches canonical continuum action (1/2) int d^4x (d hat_xi)^2 iff hat_xi = tilde_phi / (sqrt(V_0) * a)!
    a_lat_s, Ms_s, Z0_s, dphi_s = sp.symbols("a_lat M_star Z_0_D4 dphi", positive=True)
    V0_voronoi = sp.Rational(2, 1) * ((sp.Integer(1) / sp.sqrt(2)) ** 4)
    assert V0_voronoi == sp.Rational(1, 2)
    scale_c = sp.Symbol("scale_c", positive=True)  # hat_xi = scale_c * tilde_phi
    L_cont_from_lat = sp.Rational(1, 2) * ((a_lat_s ** 2) / (V0_voronoi * (a_lat_s ** 4))) * ((dphi_s / scale_c) ** 2)
    scale_sol = sp.solve(sp.Eq(L_cont_from_lat, sp.Rational(1, 2) * (dphi_s ** 2)), scale_c)[0]
    assert sp.simplify(scale_sol - sp.sqrt(2) / a_lat_s) == 0
    # Hence <hat_xi_x^2> / M_*^2 = scale_sol^2 * Z_0(D_4) / M_*^2 == 2 * Z_0(D_4) / (a * M_*)^2!
    tadpole_over_Mstar2_sym = sp.simplify((scale_sol ** 2) * Z0_s / (Ms_s ** 2))
    assert sp.simplify(tadpole_over_Mstar2_sym - 2 * Z0_s / ((a_lat_s * Ms_s) ** 2)) == 0

    # -------------------------------------------------------------------------
    # Gate 2A.2: High-Precision 4D 24-Cell (D_4) Watson Integral Z_0(D_4) (< 10^{-9} Agreement!)
    # -------------------------------------------------------------------------
    bz_means = {}
    for m_g in (24, 32, 48, 64, 96):
        th_g = (np.arange(m_g, dtype=np.float64) + 0.5) * (2.0 * math.pi / m_g) - math.pi
        cg = np.cos(th_g)
        X0, X1, X2, X3 = np.meshgrid(cg, cg, cg, cg, indexing="ij")
        Lg = 8.0 + (2.0 / 3.0) * (X0 ** 2 + X1 ** 2 + X2 ** 2 + X3 ** 2) - (2.0 / 3.0) * ((X0 + X1 + X2 + X3) ** 2)
        bz_means[m_g] = float(np.mean(1.0 / Lg))

    r_32_64 = (4.0 * bz_means[64] - bz_means[32]) / 3.0
    r_48_96 = (4.0 * bz_means[96] - bz_means[48]) / 3.0
    z0_d4_richardson = (16.0 * r_48_96 - r_32_64) / 15.0

    u_gl, w_gl = np.polynomial.legendre.leggauss(200)
    th_1d = (np.arange(256, dtype=np.float64) + 0.5) * (math.pi / 256.0)
    c_1d = np.cos(th_1d)
    omc_1d = 1.0 - c_1d
    smc_1d = 7.0 - c_1d

    def P_return_d4(sig: float) -> float:
        if sig <= 0.0:
            return 1.0
        rt = math.sqrt(sig / 3.0)
        z0 = 8.0 * rt
        u_min = -z0 if z0 < 14.0 else -14.0
        hw = 0.5 * (14.0 - u_min)
        mu_s = 0.5 * (14.0 + u_min)
        u_pts = mu_s + hw * u_gl
        w_pts = (2.0 / math.sqrt(2.0 * math.pi)) * hw * w_gl * np.exp(-0.5 * (u_pts ** 2))
        exponent = -(2.0 / 3.0) * sig * (omc_1d * smc_1d)[None, :] - 2.0 * rt * u_pts[:, None] * omc_1d[None, :]
        I1 = np.mean(np.exp(exponent), axis=1)
        return float(np.sum(w_pts * (I1 ** 4)))

    for s_chk in (0.5, 2.0):
        th_c = (np.arange(24, dtype=np.float64) + 0.5) * (2.0 * math.pi / 24.0) - math.pi
        cc = np.cos(th_c)
        Y0, Y1, Y2, Y3 = np.meshgrid(cc, cc, cc, cc, indexing="ij")
        Lc = 8.0 + (2.0 / 3.0) * (Y0 ** 2 + Y1 ** 2 + Y2 ** 2 + Y3 ** 2) - (2.0 / 3.0) * ((Y0 + Y1 + Y2 + Y3) ** 2)
        p_4d = float(np.mean(np.exp(-s_chk * Lc)))
        assert abs(P_return_d4(s_chk) - p_4d) < 1.0e-13

    R_sig = 120.0
    z0_pt_low, _ = integrate.quad(P_return_d4, 0.0, R_sig, epsabs=1.0e-13, epsrel=1.0e-13, limit=300)
    C_IR = 1.0 / (32.0 * (math.pi ** 2))
    z0_pt_tail = C_IR * (1.0 / R_sig + 1.0 / (8.0 * (R_sig ** 2)))
    z0_d4_proper_time = z0_pt_low + z0_pt_tail
    rel_diff_methods = abs(z0_d4_proper_time - z0_d4_richardson) / z0_d4_richardson
    assert rel_diff_methods < 1.0e-9, f"D_4 Watson integral mismatch > 1e-9: {rel_diff_methods:.3e}"

    # Gate 3A: Verify exact W(F_4) 12-link resolvent transitivity Pi_{mm}^{(D_4)} == (1/4) Z_0(D_4)!
    m_res = 32
    th_r = (np.arange(m_res, dtype=np.float64) + 0.5) * (2.0 * math.pi / m_res) - math.pi
    R0, R1, R2, R3 = np.meshgrid(th_r, th_r, th_r, th_r, indexing="ij")
    Qr = np.stack([R0.ravel(), R1.ravel(), R2.ravel(), R3.ravel()], axis=1) * math.sqrt(2.0)
    qe_r = mm(Qr, E12.T)
    Lr = (2.0 / 3.0) * np.sum(1.0 - np.cos(qe_r), axis=1)
    z0_r32 = float(np.mean(1.0 / Lr))
    pi_mm_links = [float(np.mean(2.0 * (1.0 - np.cos(qe_r[:, m_i])) / (Lr ** 2))) for m_i in range(12)]
    max_link_spread = max(abs(pv - 0.25 * z0_r32) for pv in pi_mm_links)
    assert max_link_spread < 1.0e-15

    # -------------------------------------------------------------------------
    # Gate 2A.3: Bare vs. Matched UV Scale Audit on 24-Cell (D_4)
    # -------------------------------------------------------------------------
    u2_val = 1.0 / 192.0
    u4_val = u2_val ** 2
    W_D4 = 2.0 * z0_d4_richardson  # G_{D_4}(0; a = 1) = Z_0(D_4) / V_0 = 2 Z_0(D_4)
    jeans_upper_bound = 2.41e-6
    void_star_lower_bound = 1.50e-6

    # (i) PROVEN NO-GO at naive bare edge identification a = l_* = M_*^{-1} (a M_* = 1):
    dc_bare_12link = 12.0 * u4_val * W_D4           # = 24 Z_0(D_4) u^4 = 8.9994e-5 (37.34x violation!)
    dc_bare_singlet = u4_val * W_D4                 # =  2 Z_0(D_4) u^4 = 7.4995e-6 ( 3.11x violation!)
    dc_sqrt3_singlet = u4_val * W_D4 / 3.0          # = (2/3) Z_0(D_4) u^4 = 2.4998e-6 (> 2.41e-6!)
    assert dc_bare_12link / jeans_upper_bound > 35.0
    assert dc_bare_singlet / jeans_upper_bound > 3.0
    assert dc_sqrt3_singlet > jeans_upper_bound

    # (ii) CONDITIONAL Candidate A: 12-link quadratic strain + heat-kernel tadpole matching G_{D_4}(0; a) == M_*^2 / (16 pi^2):
    a_times_Mstar_hk = 4.0 * math.pi * math.sqrt(W_D4)
    debye_over_Mstar_hk = math.sqrt(8.0 * math.pi) / a_times_Mstar_hk
    dc_candA_12link = 12.0 * u4_val * (W_D4 / (a_times_Mstar_hk ** 2))
    dc_3_over_4pi2 = (3.0 / (4.0 * (math.pi ** 2))) * u4_val
    mR_over_Mstar = ((0.25 ** 2) / (16.0 * (math.pi ** 2)))
    assert abs(dc_candA_12link - dc_3_over_4pi2) < 1.0e-20
    assert abs(dc_candA_12link - mR_over_Mstar * u2_val) < 1.0e-20

    k_s, mxi_s = sp.symbols("k_rg m_xi", positive=True)
    litim_init_tad = (Ms_s ** 4) / (32 * (sp.pi ** 2)) / ((Ms_s ** 2) + (mxi_s ** 2))
    litim_flow_tad = sp.integrate((k_s ** 5) / (16 * (sp.pi ** 2) * (((k_s ** 2) + (mxi_s ** 2)) ** 2)), (k_s, 0, Ms_s))
    tot_litim_tad = sp.simplify(litim_init_tad + litim_flow_tad)
    expected_cov_tad = ((Ms_s ** 2) - (mxi_s ** 2) * sp.log(1 + (Ms_s ** 2) / (mxi_s ** 2))) / (16 * (sp.pi ** 2))
    assert sp.simplify(tot_litim_tad - expected_cov_tad) == 0
    f_thr_mR = 1.0 - (mR_over_Mstar ** 2) * math.log(1.0 + 1.0 / (mR_over_Mstar ** 2))
    assert abs(f_thr_mR - 1.0) < 5.0e-6

    # (iii) CONDITIONAL Candidate B: 1-mode W(F_4) radial-breathing singlet at circumdiameter scale a = 2 l_* (a M_* = 2):
    # Exact viable window for a M_* in 1-mode singlet d_c(a M_*) = W_D4 * u^4 / (a M_*)^2 in [1.50e-6, 2.41e-6]:
    aM_singlet_min = math.sqrt(W_D4 * u4_val / jeans_upper_bound)       # 1.7641
    aM_singlet_max = math.sqrt(W_D4 * u4_val / void_star_lower_bound)   # 2.2359
    assert aM_singlet_min < 2.0 < aM_singlet_max
    dc_candB_singlet_a2 = (W_D4 / (2.0 ** 2)) * u4_val  # == (1/2) Z_0(D_4) u^4 = 1.874871e-6
    assert abs(dc_candB_singlet_a2 - 0.5 * z0_d4_richardson * u4_val) < 1.0e-22
    assert void_star_lower_bound <= dc_candB_singlet_a2 <= jeans_upper_bound

    r_grid = np.concatenate([[0.0], np.geomspace(1.0e-17, 100.0, 4000)])
    R_sun_Mpc = 6.957e8 / mod_f.MPC_M
    singlet_params = {
        "K_B": 0.05,
        "lam_s0": 50.0,
        "K_2": 100.0,
        "mu0_Mpc": 0.65,
        "d_c": dc_candB_singlet_a2,
        "d_max": u2_val,
        "alpha_lam": 2.0,
        "m_sig_Mpc": 0.65,
        "sig_c": 1.0e-13,
        "kap_sig": 2.0e8,
        "y_sig_sat": 2.0e-11,
    }
    star_sing = mod_f.solve_self_consistent_3field_schur_bvp(r_grid, 1.0, R_sun_Mpc, singlet_params)
    mw_sing = mod_f.solve_self_consistent_3field_schur_bvp(r_grid, 6.0e10, 1.0e-3, singlet_params)
    rc_sing = mod_f.solve_self_consistent_3field_schur_bvp(r_grid, 1.0e15, 0.25, singlet_params)
    for sys_r in (star_sing, mw_sing, rc_sing):
        assert sys_r["schur_varphi_flux_rel_residual"] < 1.0e-12
        assert sys_r["sigma_helmholtz_rel_residual"] < 1.0e-10
        assert sys_r["min_Schur_2x2_Jacobian"] > 0.0
        assert sys_r["max_rel_Poisson_dPsi_sigma_over_Psi"] < 1.0e-5
    assert star_sing["max_Psi_over_d_eff_all_r"] < 1.20

    return {
        "gate1a_capitani2003_Z0_Z4_exact": CAPITANI_2003_Z0_Z4,
        "gate1a_capitani2003_Z0_Z4_computed": z0_z4_num,
        "gate1a_capitani2003_abs_err": err_z0_z4,
        "gate1a_hasenfratz_karsch_Z4_susceptibility": chi_hk_z4,
        "gate1a_naive_linear_Z4_divergence": chi_naive_z4,
        "gate1a_hasenfratz_karsch_D4_susceptibility": chi_hk_d4,
        "gate1a_naive_linear_D4_divergence": chi_naive_d4,
        "theorem_K1_dmax_transverse_SO2_Casimir": u2_val,
        "theorem_K1_12link_seed_identity_12u2_eq_rseed2": True,
        "theorem_K1_HK_cosh_prefactor": str(V0_sol),
        "theorem_K1_Voronoi_V0_Jacobian_tadpole_formula": str(tadpole_over_Mstar2_sym),
        "Z0_D4_richardson_m24_to_96": z0_d4_richardson,
        "Z0_D4_proper_time_hubbard_stratonovich": z0_d4_proper_time,
        "Z0_D4_two_method_rel_diff": rel_diff_methods,
        "W_D4_continuum_norm_tadpole_2Z0": W_D4,
        "WF4_12link_resolvent_Pi_mm_max_spread": max_link_spread,
        "proven_nogo_bare_edge_a_eq_lstar": {
            "d_c_bare_12link_24Z0_u4": dc_bare_12link,
            "bare_12link_jeans_violation_factor": dc_bare_12link / jeans_upper_bound,
            "d_c_bare_singlet_2Z0_u4": dc_bare_singlet,
            "bare_singlet_jeans_violation_factor": dc_bare_singlet / jeans_upper_bound,
            "d_c_sqrt3_singlet_2over3_Z0_u4": dc_sqrt3_singlet,
            "cosmic_jeans_upper_bound": jeans_upper_bound,
            "PROVEN_NO_GO": True,
        },
        "conditional_candidate_A_12link_heat_kernel_matched": {
            "a_times_M_star": a_times_Mstar_hk,
            "Debye_Lambda_BZ_over_M_star": debye_over_Mstar_hk,
            "d_c_3_over_4pi2_u4": dc_candA_12link,
            "equals_mR_over_Mstar_times_dmax": True,
            "finite_mR_threshold_factor": f_thr_mR,
            "epistemic_status": "CONDITIONAL (algebraic Z_0 cancellation exact given 12-link strain coupling + G_{D_4}(0;a) == M_*^2/(16 pi^2) matching)",
        },
        "conditional_candidate_B_singlet_circumdiameter_a_eq_2lstar": {
            "viable_a_times_Mstar_band": [round(aM_singlet_min, 4), round(aM_singlet_max, 4)],
            "a_times_M_star_circumdiameter": 2.0,
            "d_c_half_Z0D4_u4": dc_candB_singlet_a2,
            "IsolatedVoidStar_Psi_over_dc": star_sing["max_Psi_over_d_eff_all_r"],
            "IsolatedVoidStar_cosh": math.cosh(star_sing["max_Psi_over_d_eff_all_r"]),
            "MilkyWay_max_Psi_over_deff": mw_sing["max_Psi_over_d_eff_all_r"],
            "RichCluster_max_Psi_over_deff": rc_sing["max_Psi_over_d_eff_all_r"],
            "max_3field_Schur_rel_residual": max(
                star_sing["schur_varphi_flux_rel_residual"],
                mw_sing["schur_varphi_flux_rel_residual"],
                rc_sing["schur_varphi_flux_rel_residual"],
            ),
            "epistemic_status": "CONDITIONAL (1-mode W(F_4) radial singlet at 24-cell circumdiameter scale a = 2 l_*)",
        },
    }


def derive_sympy_fastpt_tatt_and_spt_kernels() -> dict:
    """Gate 1B & Gate 2B (First-Principles SymPy Derivation):
      1. Derives F_2(q_1, q_2) and F_3^{sym}(k, q, -q) from Goroff-Grinstein-Rey-Wise (1986) fluid
         recursion, proves exact Galilean 1/r^2 IR pole cancellation, and verifies the exact
         Makino-Sasaki-Suto (1992, PRD 46, 585) power-law P_lin(q) = q^{-2} 1-loop SPT benchmark
         P_{22}(1) + P_{13}(1) == 55 / 784 = 0.0701530612!
      2. Derives ALL Blazek et al. (2019, PRD 100, 103506, Appendix A) TATT 1-loop mode-coupling AND
         1PR propagator kernels in SymPy — including BOTH P_{0|0E} = 2 A_{0|0E} + B_{0|0E}^{ren} AND
         the complete P_{0|E2} = 2 A_{0|E2} + 4 B_{0|E2}^{ren} (`ia_mix[0] + ia_mix[1]`) — proving:
         - Exact closed-form rational+log propagator integrals B_{0|0E}(r) (limit +58/105) and
           4 B_{0|E2}(r) (limit +58/315);
         - Exact Galilean IR pole cancellations in BOTH P_{0|0E} (251/105) and P_{0|E2} (32/315);
         - Exact k -> 0 momentum-conservation zero 2 A_{0|E2}(0) == 0 and E/B white-noise parity!
    """
    r, x, cphi, sphi, st = sp.symbols("r x cphi sphi st", real=True)

    def norm2(a):
        return sp.simplify(
            sp.sympify(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).subs(cphi ** 2, 1 - sphi ** 2).subs(st ** 2, 1 - x ** 2)
        )

    def add3(a, b):
        return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

    def dot3_simp(a, b):
        return sp.simplify(
            sp.sympify(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).subs(cphi ** 2, 1 - sphi ** 2).subs(st ** 2, 1 - x ** 2)
        )

    def alpha_fn(k1, k2):
        return dot3_simp(add3(k1, k2), k1) / norm2(k1)

    def beta_fn(k1, k2):
        k12 = add3(k1, k2)
        return norm2(k12) * dot3_simp(k1, k2) / (2 * norm2(k1) * norm2(k2))

    def F2_vec(q1, q2):
        return sp.Rational(5, 14) * (alpha_fn(q1, q2) + alpha_fn(q2, q1)) + sp.Rational(2, 7) * beta_fn(q1, q2)

    def G2_vec(q1, q2):
        return sp.Rational(3, 14) * (alpha_fn(q1, q2) + alpha_fn(q2, q1)) + sp.Rational(4, 7) * beta_fn(q1, q2)

    def F3_unsym(q1, q2, q3):
        q23 = add3(q2, q3)
        return (
            sp.Rational(7, 18) * alpha_fn(q1, q23) * F2_vec(q2, q3)
            + sp.Rational(7, 18) * alpha_fn(q23, q1) * G2_vec(q2, q3)
            + sp.Rational(2, 9) * beta_fn(q1, q23) * G2_vec(q2, q3)
        )

    def azim_avg_poly(expr):
        p = sp.Poly(sp.expand(expr), cphi, sphi, st)
        res = sp.Integer(0)
        for (a, b, c_st), coeff in p.terms():
            if a % 2 == 0 and b % 2 == 0:
                assert c_st % 2 == 0
                ia, ib = a // 2, b // 2
                wt = sp.factorial(2 * ia) * sp.factorial(2 * ib) / (
                    (sp.Integer(4) ** (ia + ib)) * sp.factorial(ia) * sp.factorial(ib) * sp.factorial(ia + ib)
                )
                res += coeff * wt * ((1 - x ** 2) ** (c_st // 2))
        return sp.factor(sp.simplify(res))

    kv = [sp.Integer(1), sp.Integer(0), sp.Integer(0)]
    qv_p = [r * x, r * st, sp.Integer(0)]
    mqv_p = [-r * x, -r * st, sp.Integer(0)]
    perms_4 = [(qv_p, kv, mqv_p), (qv_p, mqv_p, kv), (mqv_p, kv, qv_p), (mqv_p, qv_p, kv)]
    ker_13 = sp.factor(sp.simplify(sum(F3_unsym(a, b, c) for (a, b, c) in perms_4)).subs(st ** 2, 1 - x ** 2))

    q1 = [r * x, r * st * cphi, r * st * sphi]
    q2 = [1 - r * x, -r * st * cphi, -r * st * sphi]
    qpk = [1 + r * x, r * st * cphi, r * st * sphi]
    q1_sq = r ** 2
    q2_sq = 1 + r ** 2 - 2 * r * x
    qpk_sq = 1 + r ** 2 + 2 * r * x

    F2_rx = sp.factor(sp.simplify((3 * r + 7 * x - 10 * r * (x ** 2)) / (14 * r * q2_sq)))
    assert sp.simplify(azim_avg_poly(F2_vec(q1, q2)) - F2_rx) == 0

    ser_spt_ir = sp.simplify(sp.integrate(sp.series(4 * (F2_rx ** 2) / q2_sq + ker_13, r, 0, 3).removeO(), (x, -1, 1)) / 2)
    assert sp.simplify(ser_spt_ir.subs(r, 0) - sp.Rational(5816, 2205)) == 0

    xg, wg = np.polynomial.legendre.leggauss(64)
    ker13_np = sp.lambdify((r, x), ker_13, "numpy")
    f2_np = sp.lambdify((r, x), F2_rx, "numpy")

    def mss_folded_integrand(rv: float) -> float:
        xmax = min(1.0, 0.5 / rv)
        xm = 0.5 * (-1.0 + xmax) + 0.5 * (xmax + 1.0) * xg
        wm = 0.5 * (xmax + 1.0) * wg
        q2m = 1.0 + rv ** 2 - 2.0 * rv * xm
        i22_fold = 2.0 * 0.5 * float(np.sum(wm * 2.0 * (f2_np(rv, xm) ** 2) / q2m))
        i13 = 0.5 * float(np.sum(wg * ker13_np(rv, xg)))
        return (i22_fold + i13) / (2.0 * (math.pi ** 2))

    v1, _ = integrate.quad(mss_folded_integrand, 1.0e-5, 0.5, epsabs=1.0e-12, epsrel=1.0e-12, limit=200)
    v2, _ = integrate.quad(mss_folded_integrand, 0.5, 1.0 - 1.0e-9, epsabs=1.0e-12, epsrel=1.0e-12, limit=200)
    v3, _ = integrate.quad(mss_folded_integrand, 1.0 + 1.0e-9, 50.0, epsabs=1.0e-12, epsrel=1.0e-12, limit=200)
    v4, _ = integrate.quad(lambda u: mss_folded_integrand(1.0 / u) / (u ** 2), 1.0e-7, 0.02, epsabs=1.0e-12, epsrel=1.0e-12, limit=200)
    mss_num = v1 + v2 + v3 + v4
    mss_exact = 55.0 / 784.0
    mss_rel_err = abs(mss_num - mss_exact) / mss_exact
    assert mss_rel_err < 5.0e-5

    def s_mat(qv, qsq):
        return sp.Matrix(3, 3, lambda i, j: qv[i] * qv[j] / qsq - (sp.Rational(1, 3) if i == j else 0))

    def E_proj(T):
        return T[0, 0] - T[1, 1]

    def B_proj(T):
        return 2 * T[0, 1]

    assert E_proj(s_mat(kv, 1)) == 1 and B_proj(s_mat(kv, 1)) == 0

    S1 = s_mat(q1, q1_sq)
    S2 = s_mat(q2, q2_sq)
    Spk = s_mat(qpk, qpk_sq)

    K_0E = sp.Rational(1, 2) * (E_proj(S1) + E_proj(S2))
    K_0B = sp.Rational(1, 2) * (B_proj(S1) + B_proj(S2))

    P12 = sp.Rational(1, 2) * (S1 * S2 + S2 * S1)
    T_TT = P12 - sp.Rational(1, 3) * sp.trace(S1 * S2) * sp.eye(3)
    K_E2 = E_proj(T_TT)
    K_B2 = B_proj(T_TT)

    # Propagator coupling of delta^{(1)}(k) with [s^{(1)} s^{(2)} + s^{(2)} s^{(1)}]_E(-k):
    P_pk = sp.Rational(1, 2) * (S1 * Spk + Spk * S1)
    T_TT_pk = P_pk - sp.Rational(1, 3) * sp.trace(S1 * Spk) * sp.eye(3)

    ker_0_0E_mc = 2 * azim_avg_poly(F2_rx * K_0E)
    ker_0_0E_p1 = 2 * azim_avg_poly(F2_vec([-1, 0, 0], q1) * E_proj(S1))
    ker_0_0E_p2 = 2 * azim_avg_poly(F2_vec([1, 0, 0], q1) * E_proj(Spk))
    ker_0_0E_prop = sp.factor(sp.simplify(ker_0_0E_p1 + ker_0_0E_p2))

    ker_0_E2_mc = 2 * azim_avg_poly(F2_rx * K_E2)
    ker_0_E2_prop = 4 * azim_avg_poly(F2_vec([1, 0, 0], q1) * E_proj(T_TT_pk))

    ker_0E_0E = 2 * azim_avg_poly(K_0E ** 2)
    ker_0B_0B = 2 * azim_avg_poly(K_0B ** 2)
    ker_0E_E2 = 2 * azim_avg_poly(K_0E * K_E2)
    ker_E2_E2 = 2 * azim_avg_poly(K_E2 ** 2)
    ker_B2_B2 = 2 * azim_avg_poly(K_B2 ** 2)

    # Gate 1B & Gate 3B exact symbolic identity checks:
    ir_0_0E = sp.simplify(sp.integrate(sp.series(2 * ker_0_0E_mc + ker_0_0E_prop, r, 0, 1).removeO(), (x, -1, 1)) / 2)
    ir_0_E2 = sp.simplify(sp.integrate(sp.series(2 * ker_0_E2_mc + ker_0_E2_prop, r, 0, 1).removeO(), (x, -1, 1)) / 2)
    assert ir_0_0E == sp.Rational(251, 105)
    assert ir_0_E2 == sp.Rational(32, 315)

    lim_0_E2_mc = sp.integrate(sp.limit(ker_0_E2_mc, r, sp.oo), (x, -1, 1)) / 2
    lim_E2_E2 = sp.integrate(sp.limit(ker_E2_E2, r, sp.oo), (x, -1, 1)) / 2
    lim_B2_B2 = sp.integrate(sp.limit(ker_B2_B2, r, sp.oo), (x, -1, 1)) / 2
    lim_0E_0E = sp.integrate(sp.limit(ker_0E_0E, r, sp.oo), (x, -1, 1)) / 2
    lim_0B_0B = sp.integrate(sp.limit(ker_0B_0B, r, sp.oo), (x, -1, 1)) / 2
    lim_0E_E2 = sp.integrate(sp.limit(ker_0E_E2, r, sp.oo), (x, -1, 1)) / 2
    assert lim_0_E2_mc == 0
    assert lim_E2_E2 == sp.Rational(8, 135) and lim_B2_B2 == sp.Rational(8, 135)
    assert lim_0E_0E == sp.Rational(8, 15) and lim_0B_0B == sp.Rational(8, 15)
    assert lim_0E_E2 == sp.Rational(8, 45)

    u_pos = sp.Symbol("u_pos", positive=True)
    r_gt1 = u_pos + 1
    I_p1 = sp.simplify(sp.integrate(ker_0_0E_p1, (x, -1, 1)) / 2)
    I_p2 = sp.simplify(sp.integrate(ker_0_0E_p2.subs(r, r_gt1), (x, -1, 1)) / 2)
    assert I_p1 == sp.Rational(8, 105)
    bvs_eq_a8 = (
        sp.Rational(15, 224) * (((r_gt1 ** 2 - 1) ** 4) / (r_gt1 ** 3)) * sp.log((r_gt1 - 1) / (r_gt1 + 1))
        + (180 * (r_gt1 ** 6) - 660 * (r_gt1 ** 4) + 1516 * (r_gt1 ** 2) + 180) / (1344 * (r_gt1 ** 2))
    )
    diff_bvs = sp.expand_log((I_p2 - bvs_eq_a8).replace(sp.Pow, lambda b, e: sp.factor(b) ** e), force=True)
    assert sp.simplify(diff_bvs) == 0

    # Verify exact closed-form rational + log formula & +58/315 limit for 4 B_{0|E2}(r) (`ia_mix[1]`):
    I_e2_prop = sp.simplify(sp.integrate(ker_0_E2_prop.subs(r, r_gt1), (x, -1, 1)) / 2)
    blazek_4B_0_E2 = (
        sp.Rational(5, 224) * (((r_gt1 ** 2 - 1) ** 4) * (r_gt1 ** 2 + 1) / (r_gt1 ** 5)) * sp.log((r_gt1 - 1) / (r_gt1 + 1))
        + (225 * (r_gt1 ** 8) - 600 * (r_gt1 ** 6) + 1198 * (r_gt1 ** 4) - 600 * (r_gt1 ** 2) + 225) / (5040 * (r_gt1 ** 4))
    )
    diff_4B = sp.expand_log((I_e2_prop - blazek_4B_0_E2).replace(sp.Pow, lambda b, e: sp.factor(b) ** e), force=True)
    assert sp.simplify(diff_4B) == 0
    assert sp.limit(blazek_4B_0_E2, u_pos, sp.oo) == sp.Rational(58, 315)

    kernels_np = {
        "mc_0_0E": sp.lambdify((r, x), ker_0_0E_mc, "numpy"),
        "prop_0_0E": sp.lambdify((r, x), ker_0_0E_prop, "numpy"),
        "mc_0_E2": sp.lambdify((r, x), ker_0_E2_mc, "numpy"),
        "prop_0_E2": sp.lambdify((r, x), ker_0_E2_prop, "numpy"),
        "mc_0E_0E": sp.lambdify((r, x), ker_0E_0E, "numpy"),
        "mc_0B_0B": sp.lambdify((r, x), ker_0B_0B, "numpy"),
        "mc_0E_E2": sp.lambdify((r, x), ker_0E_E2, "numpy"),
        "mc_E2_E2": sp.lambdify((r, x), ker_E2_E2, "numpy"),
        "mc_B2_B2": sp.lambdify((r, x), ker_B2_B2, "numpy"),
    }

    return {
        "benchmarks": {
            "makino_sasaki_suto_1992_n_minus2_exact": mss_exact,
            "makino_sasaki_suto_1992_n_minus2_computed": mss_num,
            "makino_sasaki_suto_rel_err": mss_rel_err,
            "spt_galilean_IR_pole_cancelled_constant": "5816/2205",
            "tatt_P0_0E_galilean_IR_pole_cancelled_constant": "251/105",
            "tatt_P0_E2_galilean_IR_pole_cancelled_constant": "32/315",
            "blazek2015_eq_A8_B0_0E_identity_exact": True,
            "blazek2019_4B0_E2_propagator_identity_exact": True,
            "prop_0_0E_r_to_inf_C1_renorm_const": "58/105",
            "prop_0_E2_r_to_inf_C1_renorm_const": "58/315",
            "k_to_0_2A0_E2_mc_exact_zero": True,
            "k_to_0_EB_parity_E2_B2": "8/135",
            "k_to_0_EB_parity_0E_0B": "8/15",
            "k_to_0_0E_E2_limit": "8/45",
        },
        "kernels_np": kernels_np,
    }


def build_fastpt_quadrature_tables(kernels_np: dict) -> dict:
    """Precompute k-independent 2D (r, x) folded mode-coupling and C_1-renormalized 1PR propagator
    quadrature matrices so exact nonlocal 1-loop convolutions evaluate in < 50 ms per cosmology!
    """
    r_nodes = np.geomspace(1.0e-6, 1.0e6, 280)
    lnr_nodes = np.log(r_nodes)
    xg, wg = np.polynomial.legendre.leggauss(32)

    Nr = r_nodes.size
    Nx = xg.size
    xmax_r = np.minimum(1.0, 0.5 / r_nodes)
    X_fold = 0.5 * (-1.0 + xmax_r[:, None]) + 0.5 * (xmax_r[:, None] + 1.0) * xg[None, :]
    W_fold = 0.5 * (xmax_r[:, None] + 1.0) * wg[None, :]
    R_2d = np.broadcast_to(r_nodes[:, None], (Nr, Nx))
    q2_ratio_fold = np.sqrt(np.maximum(1.0 + (R_2d ** 2) - 2.0 * R_2d * X_fold, 1.0e-30))

    X_full = np.broadcast_to(xg[None, :], (Nr, Nx))
    K_prop_0_0E_ren = 0.5 * np.sum(wg[None, :] * kernels_np["prop_0_0E"](R_2d, X_full), axis=1) - (58.0 / 105.0)
    K_prop_0_E2_ren = 0.5 * np.sum(wg[None, :] * kernels_np["prop_0_E2"](R_2d, X_full), axis=1) - (58.0 / 315.0)

    return {
        "r_nodes": r_nodes,
        "lnr_nodes": lnr_nodes,
        "q2_ratio_fold": q2_ratio_fold,
        "K_0_0E_mc_f": W_fold * kernels_np["mc_0_0E"](R_2d, X_fold),
        "K_0_E2_mc_f": W_fold * kernels_np["mc_0_E2"](R_2d, X_fold),
        "K_0E_0E_f": W_fold * kernels_np["mc_0E_0E"](R_2d, X_fold),
        "K_0B_0B_f": W_fold * kernels_np["mc_0B_0B"](R_2d, X_fold),
        "K_0E_E2_f": W_fold * kernels_np["mc_0E_E2"](R_2d, X_fold),
        "K_E2_E2_f": W_fold * kernels_np["mc_E2_E2"](R_2d, X_fold),
        "K_B2_B2_f": W_fold * kernels_np["mc_B2_B2"](R_2d, X_fold),
        "K_prop_0_0E_ren": K_prop_0_0E_ren,
        "K_prop_0_E2_ren": K_prop_0_E2_ren,
    }


def compute_1loop_tatt_basis_spectra(k_grid: np.ndarray, P_lin_z0_ia: np.ndarray, qtabs: dict) -> dict:
    """Evaluate all z = 0 1-loop TATT mode-coupling & C_1-renormalized 1PR propagator spectra
    (BOTH P_{0|0E} = 2 A_{0|0E} + B_{0|0E}^{ren} AND P_{0|E2} = 2 A_{0|E2} + 4 B_{0|E2}^{ren}!)
    on k_grid via exact nonlocal 2D (r, x) quadrature.

    Returns standard unsubtracted DES-Y3 / FAST-PT spectra (preserving exact 3x3 Gram positivity at
    all k, where additive white noise is absorbed into galaxy shape noise N_noise) alongside an
    explicit Gram-preserving super-horizon high-pass filter W_ir(k) = (k/k_IR)^2 / (1 + (k/k_IR)^2)
    (k_IR = 0.005 Mpc^{-1}) sensitivity check.
    """
    lnk_grid = np.log(k_grid)
    lnP_grid = np.log(np.maximum(P_lin_z0_ia, 1.0e-90))

    def eval_Plin(q_arr: np.ndarray) -> np.ndarray:
        lq = np.log(np.maximum(q_arr, 1.0e-30))
        sl_lo = (lnP_grid[1] - lnP_grid[0]) / (lnk_grid[1] - lnk_grid[0])
        out = np.interp(lq, lnk_grid, lnP_grid)
        out = np.where(lq < lnk_grid[0], lnP_grid[0] + sl_lo * (lq - lnk_grid[0]), out)
        # Above k_max (~20 Mpc^{-1}), Gaussian exp(-(q/k_sm)^2) is < 10^{-300} -> 0.0:
        out = np.where(lq > lnk_grid[-1], -210.0, out)
        return np.exp(out)

    q_wn = np.geomspace(1.0e-5, 20.0, 800)
    P_wn = eval_Plin(q_wn)
    I_wn0 = float(np.trapezoid((q_wn ** 3) * (P_wn ** 2) / (2.0 * (math.pi ** 2)), np.log(q_wn)))

    r_nodes = qtabs["r_nodes"]
    lnr_nodes = qtabs["lnr_nodes"]
    q2_ratio_fold = qtabs["q2_ratio_fold"]

    # Vectorized evaluation over all k_grid points simultaneously:
    nk = k_grid.size
    P_0_0E_mc = np.zeros(nk, dtype=np.float64)
    P_0_E2_mc = np.zeros(nk, dtype=np.float64)
    c_prop_0_0E_ren = np.zeros(nk, dtype=np.float64)
    c_prop_0_E2_ren = np.zeros(nk, dtype=np.float64)
    P_0E_0E_raw = np.zeros(nk, dtype=np.float64)
    P_0B_0B_raw = np.zeros(nk, dtype=np.float64)
    P_0E_E2_raw = np.zeros(nk, dtype=np.float64)
    P_E2_E2_raw = np.zeros(nk, dtype=np.float64)
    P_B2_B2_raw = np.zeros(nk, dtype=np.float64)

    K_0_0E_mc_f = qtabs["K_0_0E_mc_f"]
    K_0_E2_mc_f = qtabs["K_0_E2_mc_f"]
    K_0E_0E_f = qtabs["K_0E_0E_f"]
    K_0B_0B_f = qtabs["K_0B_0B_f"]
    K_0E_E2_f = qtabs["K_0E_E2_f"]
    K_E2_E2_f = qtabs["K_E2_E2_f"]
    K_B2_B2_f = qtabs["K_B2_B2_f"]
    K_prop_0_0E_ren = qtabs["K_prop_0_0E_ren"]
    K_prop_0_E2_ren = qtabs["K_prop_0_E2_ren"]

    for ik in range(nk):
        kv = float(k_grid[ik])
        pref = (kv ** 3) / (2.0 * (math.pi ** 2))
        Pq1 = eval_Plin(kv * r_nodes)
        Pq2 = eval_Plin(kv * q2_ratio_fold)
        wt_r = (r_nodes ** 3) * Pq1

        P_0_0E_mc[ik] = pref * float(np.trapezoid(wt_r * np.sum(K_0_0E_mc_f * Pq2, axis=1), lnr_nodes))
        P_0_E2_mc[ik] = pref * float(np.trapezoid(wt_r * np.sum(K_0_E2_mc_f * Pq2, axis=1), lnr_nodes))
        c_prop_0_0E_ren[ik] = pref * float(np.trapezoid(wt_r * K_prop_0_0E_ren, lnr_nodes))
        c_prop_0_E2_ren[ik] = pref * float(np.trapezoid(wt_r * K_prop_0_E2_ren, lnr_nodes))
        P_0E_0E_raw[ik] = pref * float(np.trapezoid(wt_r * np.sum(K_0E_0E_f * Pq2, axis=1), lnr_nodes))
        P_0B_0B_raw[ik] = pref * float(np.trapezoid(wt_r * np.sum(K_0B_0B_f * Pq2, axis=1), lnr_nodes))
        P_0E_E2_raw[ik] = pref * float(np.trapezoid(wt_r * np.sum(K_0E_E2_f * Pq2, axis=1), lnr_nodes))
        P_E2_E2_raw[ik] = pref * float(np.trapezoid(wt_r * np.sum(K_E2_E2_f * Pq2, axis=1), lnr_nodes))
        P_B2_B2_raw[ik] = pref * float(np.trapezoid(wt_r * np.sum(K_B2_B2_f * Pq2, axis=1), lnr_nodes))

    P_0_0E_tot_ren = P_0_0E_mc + P_lin_z0_ia * c_prop_0_0E_ren
    P_0_E2_tot_ren = P_0_E2_mc + P_lin_z0_ia * c_prop_0_E2_ren

    assert abs(P_E2_E2_raw[0] / ((8.0 / 135.0) * I_wn0) - 1.0) < 0.02
    assert abs(P_B2_B2_raw[0] / ((8.0 / 135.0) * I_wn0) - 1.0) < 0.02
    assert abs(P_0E_0E_raw[0] / ((8.0 / 15.0) * I_wn0) - 1.0) < 0.02
    assert abs(P_0_E2_mc[0]) / ((8.0 / 135.0) * I_wn0) < 1.0e-3
    assert float(np.min(P_0E_0E_raw * P_E2_E2_raw - (P_0E_E2_raw ** 2))) >= -1.0e-12

    u_ir = (k_grid / 0.005) ** 2
    W_ir = u_ir / (1.0 + u_ir)
    W_ir_sq = W_ir ** 2

    return {
        "I_wn0": I_wn0,
        "P_0_0E_tot_ren": P_0_0E_tot_ren,
        "P_0_E2_tot_ren": P_0_E2_tot_ren,
        "P_0_0E_mc": P_0_0E_mc,
        "P_0_E2_mc": P_0_E2_mc,
        "c_prop_0_0E_ren": c_prop_0_0E_ren,
        "c_prop_0_E2_ren": c_prop_0_E2_ren,
        "P_0E_0E_raw": P_0E_0E_raw,
        "P_0B_0B_raw": P_0B_0B_raw,
        "P_0E_E2_raw": P_0E_E2_raw,
        "P_E2_E2_raw": P_E2_E2_raw,
        "P_B2_B2_raw": P_B2_B2_raw,
        "P_0_0E_mc_irfilt": P_0_0E_mc * W_ir,
        "P_0_E2_mc_irfilt": P_0_E2_mc * W_ir,
        "P_0E_0E_irfilt": P_0E_0E_raw * W_ir_sq,
        "P_0E_E2_irfilt": P_0E_E2_raw * W_ir_sq,
        "P_E2_E2_irfilt": P_E2_E2_raw * W_ir_sq,
    }


def gate2bc_fastpt_11param_euclid_and_3d_lightcone_mocks(sym_tatt: dict) -> dict:
    """Gate 1B.3 + Gate 2B + Gate 2C + Gate 3B:
      1. Exact Nonlocal 11-Parameter Euclid Stage-IV Fisher Marginalization using complete 1-loop
         FAST-PT spectra (including BOTH B_{0|0E}^{ren} and 4 B_{0|E2}^{ren}, re-evaluated nonlocally
         at every varied cosmology (Omega_{m,0}, sigma_8, n_s, h) with k_sm = 1.0 h Mpc^{-1}!):
             theta_11 = (alpha_disf, Omega_{m,0}, sigma_8, n_s, h, A_1, eta_1, b_TA, A_2, eta_2, theta_AGN).
      2. 64-Realization 24-Shell Coles-Jones (1991) Log-Normal + Resolution-Independent FFT-TATT
         Lightcone Shear Mocks on 160x160 patches (12 deg x 12 deg, ell_fund = 30.0, ell_Nyq = 1,200):
         - Applies exact Coles-Jones inverse correlation mapping xi_G = ln(1 + xi_target) per shell,
           smooths tidal fields at k_sm = 1.0 h Mpc^{-1} with exact resolution-independent 3D-to-2D
           Lagrangian variance matching (verified invariant between N_pix = 128 and N_pix = 160!),
           uses noiseless paired templates + 192 independent Hartlap-debiased GR calibration mocks
           (eliminating LOO tautologies!), and benchmarks non-Gaussian + nonlinear-TATT variance
           inflation against a paired 64-mock pure-Gaussian shell control.
    """
    src_p29 = (_P29_REPO / "scripts" / "verify_paper29_step1_theta_h0_void_optical_closure.py").read_text(encoding="utf-8")
    i0 = src_p29.index("def run_gate_s1_3_boltzmann_void_clock_master_closure(")
    i1 = src_p29.index("    res_hom_th = _full_run(H0_hom_th, 0.0, False)")
    fn_src = src_p29[i0:i1]
    fn_src = fn_src.replace('"E_opt_of_a": E_opt_of_a,', '"E_opt_of_a": E_opt_of_a, "N_clock_of_a": N_clock_of_a,', 1)
    fn_src = fn_src.replace(
        "lens_z, lens_eta, lens_weyl_lin, lens_dm_lin, lens_dm_weyl_lin = [], [], [], [], []",
        "lens_z, lens_eta, lens_weyl_lin, lens_dm_lin, lens_dm_weyl_lin = [], [], [], [], []\n"
        "        lens_a, lens_aH, lens_phi, lens_psi = [], [], [], []",
        1,
    )
    fn_src = fn_src.replace(
        "lens_weyl_lin.append((0.5 * phi_tot_weyl).copy())",
        "lens_weyl_lin.append((0.5 * phi_tot_weyl).copy())\n"
        "                lens_a.append(float(a_e))\n"
        "                lens_aH.append(float(aH_e))\n"
        "                lens_phi.append((phi + dpsi_ir_e).copy())\n"
        "                lens_psi.append(psi.copy())",
        1,
    )
    fn_src = fn_src.replace(
        '"lens_dm_weyl_lin": np.array(lens_dm_weyl_lin)[::-1],',
        '"lens_dm_weyl_lin": np.array(lens_dm_weyl_lin)[::-1],\n'
        '            "lens_a": np.array(lens_a)[::-1],\n'
        '            "lens_aH": np.array(lens_aH)[::-1],\n'
        '            "lens_phi": np.array(lens_phi)[::-1],\n'
        '            "lens_psi": np.array(lens_psi)[::-1],',
        1,
    )
    fn_src += (
        "    return dict(\n"
        "        H0_hom_th=H0_hom_th, H0_vc_A=H0_vc_A,\n"
        "        _compute_bg=_compute_bg, _build_se_table=_build_se_table,\n"
        "        _solve_boltzmann_l28=_solve_boltzmann_l28,\n"
        "        _halofit_pk_nl=_halofit_pk_nl,\n"
        "        k_cmb=k_cmb,\n"
        "    )\n"
    )
    ns_p29 = dict(p29.__dict__)
    exec(compile(fn_src, "<p29_module_k_hook>", "exec"), ns_p29)
    eng = ns_p29["run_gate_s1_3_boltzmann_void_clock_master_closure"](None)

    k_cmb = eng["k_cmb"]
    H0_vc_A = eng["H0_vc_A"]
    halofit_fn = eng["_halofit_pk_nl"]
    bg_A = eng["_compute_bg"](H0_vc_A, p29.ALPHA_OCSR, False)
    tr_A = eng["_solve_boltzmann_l28"](bg_A, eng["_build_se_table"](H0_vc_A))

    z_fine = np.concatenate([np.linspace(0.0, 1.5, 451), np.linspace(1.52, 4.0, 125)])
    base_A = b2c.build_boltzmann_and_halofit_fine(tr_A, k_cmb, p29.AS_PRESC_I, p29.NS_PRESC_I, halofit_fn, z_fine)
    k_hf = base_A["k_hf"]
    chi_A_fid = base_A["chi_fine"]

    a_dec = tr_A["lens_a"][::-1]
    eta_opt_dec = tr_A["lens_eta"][::-1]
    aH_dec = tr_A["lens_aH"][::-1]
    phi_dec = tr_A["lens_phi"][::-1]
    psi_dec = tr_A["lens_psi"][::-1]
    weyl_GR_dec = tr_A["lens_weyl_lin"][::-1]

    N_fn = bg_A["N_clock_of_a"]
    N_dec = N_fn(a_dec)
    d_eta_opt = np.diff(eta_opt_dec)
    d_tau = d_eta_opt * 0.5 * (1.0 / N_dec[:-1] + 1.0 / N_dec[1:])
    tau_dec = np.concatenate(([eta_opt_dec[0]], eta_opt_dec[0] + np.cumsum(d_tau)))

    eps_lna = 1.0e-5
    dN_dlna_dec = np.array([
        (float(N_fn(av * math.exp(eps_lna))) - float(N_fn(av * math.exp(-eps_lna)))) / (2.0 * eps_lna)
        if av * math.exp(eps_lna) <= 1.0
        else (float(N_fn(av)) - float(N_fn(av * math.exp(-eps_lna)))) / eps_lna
        for av in a_dec
    ])
    dN_dtau_dec = aH_dec * dN_dlna_dec
    c1_N = (1.0 - N_dec ** 2) / (2.0 * N_dec ** 2)
    c2_N = ((1.0 + N_dec ** 2) / (2.0 * N_dec ** 3)) * dN_dtau_dec

    Vu_cS0, dVu_cS0, _ = b2b.solve_director_Vu_exact_ode_refined(
        tau_dec, a_dec, aH_dec, phi_dec, psi_dec, k_cmb, c_S=0.0, n_sub=40
    )
    dWeyl_dec = -c1_N[:, None] * dVu_cS0 + c2_N[:, None] * Vu_cS0
    R_W_coarse_all = (1.0 + dWeyl_dec / weyl_GR_dec)[::-1]
    R_W_fine_cmb = interpolate.interp1d(base_A["z_coarse"], R_W_coarse_all[: base_A["z_coarse"].size], axis=0, kind="linear")(z_fine)

    b2d_json = json.loads((DATA_DIR / "bridge2d_dynamical_aest_phi_backreaction_kvir_results.json").read_text(encoding="utf-8"))
    modc_json = json.loads((DATA_DIR / "euclid_ia_cosmo_marginalization_and_isw.json").read_text(encoding="utf-8"))

    def _find_kvir(obj):
        if isinstance(obj, list) and len(obj) >= 5 and isinstance(obj[0], dict) and "k_vir_pi2_AeST_Mpc_inv" in obj[0]:
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                r = _find_kvir(v)
                if r is not None:
                    return r
        return None

    kvir_rows = _find_kvir(b2d_json)
    assert kvir_rows is not None
    kvir_pi2_fine = np.interp(
        z_fine,
        np.array([float(r["z"]) for r in kvir_rows]),
        np.array([float(r["k_vir_pi2_AeST_Mpc_inv"]) for r in kvir_rows]),
    )

    k_ext_hf = k_hf[k_cmb.size :]
    dRW_kvir = np.zeros((z_fine.size, k_hf.size), dtype=float)
    for jz in range(z_fine.size):
        r_top = float(R_W_fine_cmb[jz, -1])
        r_row = np.concatenate([R_W_fine_cmb[jz], np.full_like(k_ext_hf, r_top)])
        kv = float(kvir_pi2_fine[jz])
        dRW_kvir[jz] = np.exp(-((k_hf / kv) ** 2)) * (r_row - 1.0)

    surveys = b2c.build_survey_catalogs(z_fine)
    euc_spec = surveys["Euclid_S4"]
    nb = len(euc_spec["bins"])
    nz_bins = [b[1] for b in euc_spec["bins"]]
    f_sky_euc = euc_spec["f_sky"]
    N_noise_euc = euc_spec["N_noise"]
    ell_eval = np.geomspace(15.0, 1600.0, 56)

    h_fid = H0_vc_A / 100.0
    Om_m0_fid = base_A["Om_m0"]
    sig8_fid = base_A["sigma8_wl0"]
    ns_fid = p29.NS_PRESC_I
    om_b_phys_fid = tr_A["Om_b"] * (h_fid ** 2)
    f_nu_fid = tr_A["Om_nu_m0"] / Om_m0_fid

    z_sub = base_A["z_coarse"]
    dm_w_l = tr_A["lens_dm_weyl_lin"][: z_sub.size]
    k_eq_eh = 7.46e-2 * p29.OMEGA_M_PHYS_TAU * ((p29.T0_CMB / 2.7) ** (-2))

    def _T_nw(kv):
        q_e = kv / (13.41 * k_eq_eh)
        L0 = np.log(math.e + 1.8 * q_e)
        C0 = 14.2 + 386.0 / (1.0 + 69.9 * (q_e ** 1.08))
        return L0 / (L0 + C0 * (q_e ** 2))

    rat_dm_ext = ((k_ext_hf / float(k_cmb[-1])) ** 2) * (_T_nw(k_ext_hf) / float(_T_nw(float(k_cmb[-1]))))
    PR_fid = p29.AS_PRESC_I * ((k_hf / 0.05) ** (ns_fid - 1.0))
    lnP_mw_lin_sub = np.zeros((z_sub.size, k_hf.size), dtype=float)
    for jz in range(z_sub.size):
        dm_w_full = np.concatenate([dm_w_l[jz], float(dm_w_l[jz, -1]) * rat_dm_ext])
        Pk_lin_j = (2.0 * (math.pi ** 2) / (k_hf ** 3)) * PR_fid * (dm_w_full ** 2)
        lnP_mw_lin_sub[jz] = np.log(np.maximum(Pk_lin_j, 1.0e-80))
    lnP_mw_lin_fid = interpolate.interp1d(z_sub, lnP_mw_lin_sub, axis=0, kind="linear")(z_fine)

    def build_eh_lin_pk(om_m0: float, sig8: float, ns_val: float, h_val: float):
        om_r0 = (p29._OMEGA_G_TAU + p29._OMEGA_NU_ML_TAU + p29._OMEGA_NU_1R_TAU) / (h_val ** 2)
        om_de0 = 1.0 - om_m0 - om_r0
        Ez = np.sqrt(om_r0 * ((1.0 + z_fine) ** 4) + om_m0 * ((1.0 + z_fine) ** 3) + om_de0)
        c_over_H0 = p29.C_KMS / (100.0 * h_val)
        dz = np.diff(z_fine)
        invE = 1.0 / Ez
        chi_g = np.concatenate([[0.0], c_over_H0 * np.cumsum(0.5 * (invE[:-1] + invE[1:]) * dz)])
        D_norm = mod_c.compute_flat_lcdm_growth_D_norm(z_fine, om_m0, om_r0)
        T_eh = mod_c.eisenstein_hu_no_wiggle_transfer(k_hf, om_m0 * (h_val ** 2), om_b_phys_fid)
        P0_unnorm = (k_hf ** ns_val) * (T_eh ** 2)
        x8 = k_hf * (8.0 / h_val)
        W8 = 3.0 * (np.sin(x8) - x8 * np.cos(x8)) / (x8 ** 3)
        sig8_sq_raw = float(np.trapezoid((k_hf ** 3) * P0_unnorm / (2.0 * (math.pi ** 2)) * (W8 ** 2), np.log(k_hf)))
        P0_norm = P0_unnorm * ((sig8 ** 2) / sig8_sq_raw)
        lnP_eh = np.log(np.maximum((D_norm[:, None] ** 2) * P0_norm[None, :], 1.0e-80))
        return chi_g, Ez, D_norm, lnP_eh

    chi_eh_fid, _, _, lnP_eh_fid = build_eh_lin_pk(Om_m0_fid, sig8_fid, ns_fid, h_fid)
    chi_ratio_vcA = np.where(chi_eh_fid > 0.0, chi_A_fid / np.maximum(chi_eh_fid, 1.0e-30), 1.0)
    delta_lnP_boltz_calib = lnP_mw_lin_fid - lnP_eh_fid

    # Precompute k-independent 2D (r, x) FAST-PT quadrature tables once:
    qtabs = build_fastpt_quadrature_tables(sym_tatt["kernels_np"])

    C1_RHO_CRIT0 = 0.0134
    Z0_PIVOT_IA = 0.62
    K_BARY_PIVOT = 0.35
    nz_norm_bins = [nz_f / float(np.trapezoid(nz_f, z_fine)) for nz_f in nz_bins]
    _cosmo_cache: dict[tuple[float, float, float, float], tuple] = {}

    def get_cosmo_halofit_bundle(om_m0: float, sig8: float, ns_val: float, h_val: float) -> tuple:
        key = (round(om_m0, 10), round(sig8, 10), round(ns_val, 10), round(h_val, 10))
        if key in _cosmo_cache:
            return _cosmo_cache[key]
        chi_raw, Ez, D_norm, lnP_eh = build_eh_lin_pk(om_m0, sig8, ns_val, h_val)
        chi_u = chi_raw * chi_ratio_vcA
        chi_u[0] = 0.0
        dchi_u = np.gradient(chi_u, z_fine)
        dz_dchi_u = 1.0 / np.maximum(dchi_u, 1.0e-30)

        lnP_lin = lnP_eh + delta_lnP_boltz_calib
        om_r0 = (p29._OMEGA_G_TAU + p29._OMEGA_NU_ML_TAU + p29._OMEGA_NU_1R_TAU) / (h_val ** 2)
        lnP_nl = np.zeros_like(lnP_lin)
        for jz in range(z_fine.size):
            av = 1.0 / (1.0 + float(z_fine[jz]))
            ez2 = float(Ez[jz]) ** 2
            om_mz = (om_m0 / (av ** 3)) / ez2
            om_dez = max(1.0 - om_mz - (om_r0 / (av ** 4)) / ez2, 0.0)
            pk_nl = halofit_fn(k_hf, np.exp(lnP_lin[jz]), om_mz, om_dez, f_nu_fid)
            lnP_nl[jz] = np.log(np.maximum(pk_nl, 1.0e-80))

        H0_over_c = (100.0 * h_val) / p29.C_KMS
        pref_WG = 1.5 * om_m0 * (H0_over_c ** 2) * (1.0 + z_fine)
        g_bins = [b2c.compute_lensing_efficiency_g(chi_u, z_fine, nz_f) for nz_f in nz_bins]
        WG_bins = [pref_WG * gb for gb in g_bins]

        # Re-evaluate the EXACT nonlocal 1-loop FAST-PT convolutions at this cosmology (Omega_m0, sig8, n_s, h)
        # with Lagrangian halo smoothing scale k_sm = 1.0 * h_val Mpc^{-1} (Blazek, Vlah & Seljak 2015, Sec. 2.2)!
        k_sm_cosmo = 1.0 * h_val
        P_lin_z0_ia = np.exp(lnP_lin[0]) * np.exp(-((k_hf / k_sm_cosmo) ** 2))
        fastpt_cosmo = compute_1loop_tatt_basis_spectra(k_hf, P_lin_z0_ia, qtabs)

        bundle = (chi_u, dz_dchi_u, D_norm, lnP_nl, WG_bins, fastpt_cosmo)
        _cosmo_cache[key] = bundle
        return bundle

    def evaluate_euclid_Cell_11param_fastpt(
        alpha_disf: float,
        om_m0: float,
        sig8: float,
        ns_val: float,
        h_val: float,
        A1_ia: float,
        eta1_ia: float,
        b_ta: float,
        A2_tt: float,
        eta2_tt: float,
        theta_agn: float,
        dRW_grid: np.ndarray,
        apply_ir_highpass: bool = False,
    ) -> np.ndarray:
        """Full 11-parameter Euclid cosmic-shear + exact 1-loop FAST-PT TATT (A1, eta1, b_TA, A2, eta2) +
        baryonic AGN suppression C_ell^{ij}.
        Uses the DES-Y3 TATT amplitude convention (Secco et al. 2022, PRD 105, 023515, Eq. 12; `pyccl`
        `Om_m2_for_c2=False`: C_2(z) = 5 A_2 C_1^{crit} (Omega_m / D(z)^2) [(1+z)/(1+z_0)]^{eta_2})
        and the manifestly Gram-positive 1PR-resummed hybrid completion of Blazek et al. (2019, Eqs. 23-25):
            W_{1,eff} = W_1 + W_{1 delta} c_{prop}^{0|0E,ren} + W_2 c_{prop}^{0|E2,ren}.
        """
        chi_u, dz_dchi_u, D_norm, lnP_nl, WG_bins, fpt = get_cosmo_halofit_bundle(
            om_m0, sig8, ns_val, h_val
        )

        C1_z = -A1_ia * C1_RHO_CRIT0 * (om_m0 / D_norm) * (((1.0 + z_fine) / (1.0 + Z0_PIVOT_IA)) ** eta1_ia)
        C1d_z = b_ta * C1_z
        C2_z = 5.0 * A2_tt * C1_RHO_CRIT0 * (om_m0 / (D_norm ** 2)) * (((1.0 + z_fine) / (1.0 + Z0_PIVOT_IA)) ** eta2_tt)

        WI1_bins = [C1_z * nz_n * dz_dchi_u for nz_n in nz_norm_bins]
        WI1d_bins = [C1d_z * nz_n * dz_dchi_u for nz_n in nz_norm_bins]
        WI2_bins = [C2_z * nz_n * dz_dchi_u for nz_n in nz_norm_bins]

        c_prop_0E_base = fpt["c_prop_0_0E_ren"]
        c_prop_E2_base = fpt["c_prop_0_E2_ren"]
        p0_0E_mc_base = fpt["P_0_0E_mc_irfilt"] if apply_ir_highpass else fpt["P_0_0E_mc"]
        p0_E2_mc_base = fpt["P_0_E2_mc_irfilt"] if apply_ir_highpass else fpt["P_0_E2_mc"]
        p0E_0E_base = fpt["P_0E_0E_irfilt"] if apply_ir_highpass else fpt["P_0E_0E_raw"]
        p0E_E2_base = fpt["P_0E_E2_irfilt"] if apply_ir_highpass else fpt["P_0E_E2_raw"]
        pE2_E2_base = fpt["P_E2_E2_irfilt"] if apply_ir_highpass else fpt["P_E2_E2_raw"]

        RW_eff = 1.0 + alpha_disf * dRW_grid
        lp = ell_eval + 0.5
        lnk = np.log(k_hf)
        n_c = chi_u.size
        ne = ell_eval.size

        P_nl_ell = np.zeros((n_c, ne), dtype=float)
        c_prop_0E_ell = np.zeros((n_c, ne), dtype=float)
        c_prop_E2_ell = np.zeros((n_c, ne), dtype=float)
        P_0_0E_mc_ell = np.zeros((n_c, ne), dtype=float)
        P_0_E2_mc_ell = np.zeros((n_c, ne), dtype=float)
        P_0E_0E_ell = np.zeros((n_c, ne), dtype=float)
        P_0E_E2_ell = np.zeros((n_c, ne), dtype=float)
        P_E2_E2_ell = np.zeros((n_c, ne), dtype=float)
        RW_at_ell = np.ones((n_c, ne), dtype=float)

        z_bary_evol = (1.0 + z_fine) ** (-0.5)
        for jc in range(1, n_c):
            ch = float(chi_u[jc])
            inv_ch2 = 1.0 / (ch ** 2)
            d2_z = float(D_norm[jc]) ** 2
            d4_z = d2_z ** 2
            k_eval = lp / ch
            lk_eval = np.log(k_eval)

            uk2_b = (k_eval / K_BARY_PIVOT) ** 2
            B_bary = np.maximum(1.0 - theta_agn * float(z_bary_evol[jc]) * (uk2_b / (1.0 + uk2_b)), 0.1)
            wt_1loop = d4_z * inv_ch2 * B_bary

            p_dmo = np.exp(np.interp(lk_eval, lnk, lnP_nl[jc], left=-180.0, right=-180.0)) * inv_ch2
            P_nl_ell[jc] = p_dmo * B_bary
            c_prop_0E_ell[jc] = d2_z * np.interp(lk_eval, lnk, c_prop_0E_base, left=0.0, right=0.0)
            c_prop_E2_ell[jc] = d2_z * np.interp(lk_eval, lnk, c_prop_E2_base, left=0.0, right=0.0)
            P_0_0E_mc_ell[jc] = wt_1loop * np.interp(lk_eval, lnk, p0_0E_mc_base, left=0.0, right=0.0)
            P_0_E2_mc_ell[jc] = wt_1loop * np.interp(lk_eval, lnk, p0_E2_mc_base, left=0.0, right=0.0)
            P_0E_0E_ell[jc] = wt_1loop * np.interp(lk_eval, lnk, p0E_0E_base, left=0.0, right=0.0)
            P_0E_E2_ell[jc] = wt_1loop * np.interp(lk_eval, lnk, p0E_E2_base, left=0.0, right=0.0)
            P_E2_E2_ell[jc] = wt_1loop * np.interp(lk_eval, lnk, pE2_E2_base, left=0.0, right=0.0)
            RW_at_ell[jc] = np.interp(lk_eval, lnk, RW_eff[jc], left=float(RW_eff[jc, 0]), right=float(RW_eff[jc, -1]))

        C_tot = np.zeros((nb, nb, ne), dtype=float)
        for i in range(nb):
            WG_i = RW_at_ell * WG_bins[i][:, None]
            W1d_i = WI1d_bins[i][:, None]
            W2_i = WI2_bins[i][:, None]
            W1_eff_i = WI1_bins[i][:, None] + W1d_i * c_prop_0E_ell + W2_i * c_prop_E2_ell
            for j in range(i, nb):
                WG_j = RW_at_ell * WG_bins[j][:, None]
                W1d_j = WI1d_bins[j][:, None]
                W2_j = WI2_bins[j][:, None]
                W1_eff_j = WI1_bins[j][:, None] + W1d_j * c_prop_0E_ell + W2_j * c_prop_E2_ell

                Wtot_i = WG_i + W1_eff_i
                Wtot_j = WG_j + W1_eff_j
                term_nl = (Wtot_i * Wtot_j) * P_nl_ell
                term_0_0E_mc = (Wtot_i * W1d_j + W1d_i * Wtot_j) * P_0_0E_mc_ell
                term_0_E2_mc = (Wtot_i * W2_j + W2_i * Wtot_j) * P_0_E2_mc_ell
                term_0E_0E = (W1d_i * W1d_j) * P_0E_0E_ell
                term_0E_E2 = (W1d_i * W2_j + W2_i * W1d_j) * P_0E_E2_ell
                term_E2_E2 = (W2_i * W2_j) * P_E2_E2_ell

                val = np.trapezoid(
                    term_nl + term_0_0E_mc + term_0_E2_mc + term_0E_0E + term_0E_E2 + term_E2_E2, chi_u, axis=0
                )
                C_tot[i, j] = C_tot[j, i] = val
        return C_tot

    param_names_11 = [
        "alpha_disf", "Omega_m0", "sigma_8", "n_s", "h",
        "A_1_TA", "eta_1_TA", "b_TA", "A_2_TT", "eta_2_TT", "theta_AGN",
    ]
    steps_11 = [0.05, 0.003, 0.004, 0.005, 0.004, 0.05, 0.05, 0.05, 0.05, 0.05, 0.015]
    F_prior_planck_11 = np.diag([
        0.0,
        1.0 / (0.0073 ** 2),
        1.0 / (0.0060 ** 2),
        1.0 / (0.0042 ** 2),
        1.0 / (0.0054 ** 2),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])

    def build_11param_fisher(theta_fid: list[float], ell_min: int, ell_max: int, apply_ir_highpass: bool = False) -> np.ndarray:
        C_fid = evaluate_euclid_Cell_11param_fastpt(*theta_fid, dRW_grid=dRW_kvir, apply_ir_highpass=apply_ir_highpass)
        dC_list = []
        for p_i in range(11):
            th_p = list(theta_fid)
            th_m = list(theta_fid)
            th_p[p_i] += steps_11[p_i]
            th_m[p_i] -= steps_11[p_i]
            Cp = evaluate_euclid_Cell_11param_fastpt(*th_p, dRW_grid=dRW_kvir, apply_ir_highpass=apply_ir_highpass)
            Cm = evaluate_euclid_Cell_11param_fastpt(*th_m, dRW_grid=dRW_kvir, apply_ir_highpass=apply_ir_highpass)
            dC_list.append((Cp - Cm) / (2.0 * steps_11[p_i]))

        ell_dense = np.arange(ell_min, ell_max + 1, dtype=float)
        ll_e = np.log(ell_eval)
        ll_d = np.log(ell_dense)
        C_obs_d = np.zeros((ell_dense.size, nb, nb), dtype=float)
        for i in range(nb):
            for j in range(i, nb):
                cv = np.interp(ll_d, ll_e, C_fid[i, j])
                if i == j:
                    cv = cv + N_noise_euc
                C_obs_d[:, i, j] = C_obs_d[:, j, i] = cv
        assert float(np.min(np.linalg.eigvalsh(C_obs_d))) > 0.0, "C_obs_d lost positive-definiteness!"
        C_inv_d = np.linalg.inv(C_obs_d)

        dC_d = np.zeros((11, ell_dense.size, nb, nb), dtype=float)
        for p_i in range(11):
            for i in range(nb):
                for j in range(i, nb):
                    dv = np.interp(ll_d, ll_e, dC_list[p_i][i, j])
                    dC_d[p_i, :, i, j] = dC_d[p_i, :, j, i] = dv

        A_mats = np.einsum("lij,pljk->plik", C_inv_d, dC_d)
        mode_traces = np.einsum("plij,qlji->pql", A_mats, A_mats)
        weights = 0.5 * f_sky_euc * (2.0 * ell_dense + 1.0)
        return np.einsum("pql,l->pq", mode_traces, weights)

    # Gate 1B.3 Benchmark: Exact reproduction of Module C's 7-parameter NLA SNRs on ell in [30, 1000]
    theta_nla_limit = [0.0, Om_m0_fid, sig8_fid, ns_fid, h_fid, 1.72, -0.41, 0.0, 0.0, 0.0, 0.0]
    F_11_at_nla = build_11param_fisher(theta_nla_limit, 30, 1000, apply_ir_highpass=False)
    idx_7 = [0, 1, 2, 3, 4, 5, 6]
    F_7_sub = F_11_at_nla[np.ix_(idx_7, idx_7)]
    F_7_prior = F_prior_planck_11[np.ix_(idx_7, idx_7)]
    snr_7_euc = 1.0 / math.sqrt(float(np.linalg.inv(F_7_sub)[0, 0]))
    snr_7_plk = 1.0 / math.sqrt(float(np.linalg.inv(F_7_sub + F_7_prior)[0, 0]))

    ref_7p = modc_json["part1_euclid_7param_fisher_forecasts"]["kvir_pi2_AeST_z_EuclidIA_fid1p72_ell30_1000"]
    modc_snr_euc = float(ref_7p["SNR_7param_Euclid_alone_full_marg"])
    modc_snr_plk = float(ref_7p["SNR_7param_Euclid_plus_Planck18_full_marg"])
    assert abs(round(snr_7_euc, 2) - modc_snr_euc) <= 0.02
    assert abs(round(snr_7_plk, 2) - modc_snr_plk) <= 0.02

    # Full 11-parameter FAST-PT forecasts (standard unsubtracted DES-Y3 TATT convention as primary,
    # plus IR high-pass k_IR = 0.005 Mpc^{-1} sensitivity check):
    theta_11_high = [0.0, Om_m0_fid, sig8_fid, ns_fid, h_fid, 1.72, -0.41, 1.0, -1.36, 0.0, 0.12]
    theta_11_des = [0.0, Om_m0_fid, sig8_fid, ns_fid, h_fid, 0.70, -0.41, 1.0, -1.36, 0.0, 0.12]

    def summarize_11param(F11: np.ndarray) -> dict:
        evals11 = np.linalg.eigvalsh(F11)
        assert float(np.min(evals11)) > 0.0, f"F11 not positive definite: min_eig={np.min(evals11)}"

        def sub_snr(idxs: list[int], prior: np.ndarray | None = None) -> float:
            Fs = F11[np.ix_(idxs, idxs)].copy()
            if prior is not None:
                Fs += prior[np.ix_(idxs, idxs)]
            return 1.0 / math.sqrt(float(np.linalg.inv(Fs)[0, 0]))

        snr_1p = math.sqrt(float(F11[0, 0]))
        snr_7p_nla = sub_snr([0, 1, 2, 3, 4, 5, 6])
        snr_10p_no_bta_euc = sub_snr([0, 1, 2, 3, 4, 5, 6, 8, 9, 10])
        snr_10p_no_bta_plk = sub_snr([0, 1, 2, 3, 4, 5, 6, 8, 9, 10], F_prior_planck_11)
        snr_11p_euc = sub_snr(list(range(11)))
        snr_11p_plk = sub_snr(list(range(11)), F_prior_planck_11)

        assert snr_1p >= snr_7p_nla - 1.0e-9
        assert snr_7p_nla >= snr_10p_no_bta_euc - 1.0e-9
        assert snr_10p_no_bta_euc >= snr_11p_euc - 1.0e-9
        assert snr_10p_no_bta_plk >= snr_11p_plk - 1.0e-9
        assert snr_11p_plk >= snr_11p_euc - 1.0e-9

        Cov11 = np.linalg.inv(F11)
        sig11 = np.sqrt(np.diag(Cov11))
        corrs11 = {
            param_names_11[k]: round(float(Cov11[0, k] / (sig11[0] * sig11[k])), 4)
            for k in range(1, 11)
        }
        return {
            "SNR_1param_fixed": round(snr_1p, 2),
            "SNR_7param_cosmo_plus_NLA": round(snr_7p_nla, 2),
            "SNR_10param_FASTPT_bTA_fixed_Euclid_alone": round(snr_10p_no_bta_euc, 2),
            "SNR_10param_FASTPT_bTA_fixed_Euclid_plus_Planck18": round(snr_10p_no_bta_plk, 2),
            "SNR_11param_FASTPT_full_Euclid_alone": round(snr_11p_euc, 2),
            "SNR_11param_FASTPT_full_Euclid_plus_Planck18": round(snr_11p_plk, 2),
            "sigma_alpha_disf_11param_Euclid_alone": round(1.0 / snr_11p_euc, 4),
            "sigma_alpha_disf_11param_Euclid_plus_Planck18": round(1.0 / snr_11p_plk, 4),
            "correlations_with_alpha_disf_Euclid_alone": corrs11,
        }

    F11_high_1500 = build_11param_fisher(theta_11_high, 30, 1500, apply_ir_highpass=False)
    F11_high_1000 = build_11param_fisher(theta_11_high, 30, 1000, apply_ir_highpass=False)
    F11_high_1000_irhp = build_11param_fisher(theta_11_high, 30, 1000, apply_ir_highpass=True)
    F11_des_1000 = build_11param_fisher(theta_11_des, 30, 1000, apply_ir_highpass=False)

    res_11_high_1500 = summarize_11param(F11_high_1500)
    res_11_high_1000 = summarize_11param(F11_high_1000)
    res_11_high_1000_irhp = summarize_11param(F11_high_1000_irhp)
    res_11_des_1000 = summarize_11param(F11_des_1000)

    assert res_11_high_1500["SNR_11param_FASTPT_full_Euclid_plus_Planck18"] > 4.0
    assert res_11_high_1000["SNR_11param_FASTPT_full_Euclid_plus_Planck18"] > 3.5
    assert res_11_des_1000["SNR_11param_FASTPT_full_Euclid_plus_Planck18"] > 3.5

    chi_u, dz_dchi_u, D_norm, lnP_nl, WG_bins, fpt_fid = get_cosmo_halofit_bundle(Om_m0_fid, sig8_fid, ns_fid, h_fid)

    # -------------------------------------------------------------------------
    # Gate 2C & Gate 3B: 64-Realization 24-Shell Coles-Jones (1991) Log-Normal + FFT-TATT Lightcones
    # -------------------------------------------------------------------------
    # Grid specification: 160 x 160 flat-sky patches covering 12 deg x 12 deg (Omega_patch = 144 deg^2):
    #   - Fundamental mode: ell_fund = 360 / 12 = 30.0 (exact match to ell_min = 30!);
    #   - Inscribed Nyquist circle: ell_Nyq = 180 * 80 / 12 = 1,200.0 > 1,000 (100% unclipped annuli!).
    N_pix = 160
    L_deg = 12.0
    L_rad = math.radians(L_deg)
    Omega_patch = L_rad ** 2
    dx_rad = L_rad / N_pix

    z_edges = np.linspace(0.05, 2.50, 25)
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    chi_edges = np.interp(z_edges, z_fine, chi_u)
    dchi_sh = np.diff(chi_edges)
    chi_sh = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    D_sh = np.interp(z_mid, z_fine, D_norm)
    n_sh = z_mid.size

    WG_sh = np.array([np.interp(z_mid, z_fine, WG_bins[b_i]) for b_i in range(nb)])
    nz_sh = np.array([np.interp(z_mid, z_fine, nz_norm_bins[b_i] * dz_dchi_u) for b_i in range(nb)])

    C1_sh = -1.72 * C1_RHO_CRIT0 * (Om_m0_fid / D_sh) * (((1.0 + z_mid) / (1.0 + Z0_PIVOT_IA)) ** (-0.41))
    C1d_sh = 1.0 * C1_sh
    C2_sh = 5.0 * (-1.36) * C1_RHO_CRIT0 * (Om_m0_fid / (D_sh ** 2)) * (((1.0 + z_mid) / (1.0 + Z0_PIVOT_IA)) ** 0.0)

    # Exact analytical continuum 3D-to-2D Limber radial correlation-cell factor per shell (zero grid dependence!):
    #   sigma_{3D,sm}^2(z_s) = int_0^inf [dk k^2 / (2 pi^2)] P_m(k, z_s) B_b(k, z_s) exp(-(k/k_sm)^2)
    #   sigma_{2D,sm}^2(z_s) = (1 / dchi_s) int_0^inf [dk k / (2 pi)] P_m(k, z_s) B_b(k, z_s) exp(-(k/k_sm)^2)
    #   => lambda_s = sigma_{3D,sm}(z_s) / sigma_{2D,sm}(z_s) == sqrt((dchi_s / pi) * <k>_{sm}(z_s))!
    k_sm_fid = 1.0 * h_fid
    W_sm_1d_sq = np.exp(-((k_hf / k_sm_fid) ** 2))
    P_lin_z0_sm = np.exp(lnP_mw_lin_fid[0]) * W_sm_1d_sq
    sig_3d_ia_z0 = math.sqrt(float(np.trapezoid((k_hf ** 3) * P_lin_z0_sm / (2.0 * (math.pi ** 2)), np.log(k_hf))))

    lnk = np.log(k_hf)
    lambda_3d_2d_exact_sh = np.zeros(n_sh, dtype=np.float64)
    sig_2d_cont_sh = np.zeros(n_sh, dtype=np.float64)
    for is_h in range(n_sh):
        dc_s = float(dchi_sh[is_h])
        zv_s = float(z_mid[is_h])
        jz_near = int(np.argmin(np.abs(z_fine - zv_s)))
        uk2_1d = (k_hf / K_BARY_PIVOT) ** 2
        B_b_1d = np.maximum(1.0 - 0.12 * ((1.0 + zv_s) ** (-0.5)) * (uk2_1d / (1.0 + uk2_1d)), 0.1)
        P_sm_1d = np.exp(lnP_nl[jz_near]) * B_b_1d * W_sm_1d_sq
        var_3d_s = float(np.trapezoid((k_hf ** 3) * P_sm_1d / (2.0 * (math.pi ** 2)), lnk))
        var_2d_s = float(np.trapezoid((k_hf ** 2) * P_sm_1d / (2.0 * math.pi), lnk)) / dc_s
        sig_2d_cont_sh[is_h] = math.sqrt(var_2d_s)
        lambda_3d_2d_exact_sh[is_h] = math.sqrt(var_3d_s / var_2d_s)

    def build_shell_grid_operators(n_pix_grid: int) -> dict:
        dx_g = L_rad / n_pix_grid
        lx_g = 2.0 * math.pi * np.fft.fftfreq(n_pix_grid, d=dx_g)
        LX_g, LY_g = np.meshgrid(lx_g, lx_g, indexing="ij")
        Lm_g = np.hypot(LX_g, LY_g)
        Lsq_g = np.where(Lm_g > 0.0, Lm_g ** 2, 1.0)

        C_target_sh = np.zeros((n_sh, n_pix_grid, n_pix_grid), dtype=np.float64)
        C_gauss_cj_sh = np.zeros((n_sh, n_pix_grid, n_pix_grid), dtype=np.float64)
        var_gauss_cj_sh = np.zeros(n_sh, dtype=np.float64)
        W_sm_2d_sh = np.zeros((n_sh, n_pix_grid, n_pix_grid), dtype=np.float64)
        sig_2d_grid_sh = np.zeros(n_sh, dtype=np.float64)
        RW_2d_sh = np.ones((n_sh, n_pix_grid, n_pix_grid), dtype=np.float64)

        for is_h in range(n_sh):
            ch_s = float(chi_sh[is_h])
            dc_s = float(dchi_sh[is_h])
            zv_s = float(z_mid[is_h])
            jz_near = int(np.argmin(np.abs(z_fine - zv_s)))
            k_2d = np.maximum(Lm_g / ch_s, 1.0e-5)
            lk_2d = np.log(k_2d)
            p_3d = np.exp(np.interp(lk_2d, lnk, lnP_nl[jz_near], left=-180.0, right=-180.0))
            uk2_b = (k_2d / K_BARY_PIVOT) ** 2
            B_b = np.maximum(1.0 - 0.12 * ((1.0 + zv_s) ** (-0.5)) * (uk2_b / (1.0 + uk2_b)), 0.1)
            c_del = np.where(Lm_g > 0.0, p_3d * B_b / (dc_s * (ch_s ** 2)), 0.0)
            C_target_sh[is_h] = c_del

            xi_target = ((n_pix_grid ** 2) / Omega_patch) * np.fft.ifft2(c_del).real
            xi_G = np.log(np.maximum(1.0 + xi_target, 1.0e-12))
            c_g = np.maximum((Omega_patch / (n_pix_grid ** 2)) * np.fft.fft2(xi_G).real, 0.0)
            c_g[0, 0] = 0.0
            C_gauss_cj_sh[is_h] = c_g
            var_gauss_cj_sh[is_h] = float(np.sum(c_g)) / Omega_patch

            w_sm = np.where(Lm_g > 0.0, np.exp(-0.5 * ((k_2d / k_sm_fid) ** 2)), 0.0)
            W_sm_2d_sh[is_h] = w_sm
            sig_2d_grid_sh[is_h] = math.sqrt(max(float(np.sum(c_del * (w_sm ** 2))) / Omega_patch, 1.0e-30))

            rw_1d = 1.0 + dRW_kvir[jz_near]
            RW_2d_sh[is_h] = np.where(Lm_g > 0.0, np.interp(lk_2d, lnk, rw_1d, left=float(rw_1d[0]), right=float(rw_1d[-1])), 1.0)

        return {
            "N_pix": n_pix_grid,
            "dx_rad": dx_g,
            "L_mag": Lm_g,
            "cos2psi": np.where(Lm_g > 0.0, ((LX_g ** 2) - (LY_g ** 2)) / Lsq_g, 0.0),
            "sin2psi": np.where(Lm_g > 0.0, (2.0 * LX_g * LY_g) / Lsq_g, 0.0),
            "kxhat2_m_13": np.where(Lm_g > 0.0, (LX_g ** 2) / Lsq_g - (1.0 / 3.0), 0.0),
            "kyhat2_m_13": np.where(Lm_g > 0.0, (LY_g ** 2) / Lsq_g - (1.0 / 3.0), 0.0),
            "kxhyh": np.where(Lm_g > 0.0, (LX_g * LY_g) / Lsq_g, 0.0),
            "C_target_sh": C_target_sh,
            "C_gauss_cj_sh": C_gauss_cj_sh,
            "var_gauss_cj_sh": var_gauss_cj_sh,
            "W_sm_2d_sh": W_sm_2d_sh,
            "sig_2d_grid_sh": sig_2d_grid_sh,
            "lambda_3d_2d_sh": lambda_3d_2d_exact_sh.copy(),
            "RW_2d_sh": RW_2d_sh,
        }

    # Gate 3B.1: Assert BOTH exact N_pix independence of analytical lambda_3d_2d_exact_sh AND < 1.5%
    # convergence of discrete 2D Parseval sum sig_2d_grid_sh on well-resolved low-z shells (z <= 0.45, where ell_Nyq > 2 * chi * k_sm)!
    ops_128 = build_shell_grid_operators(128)
    ops_160 = build_shell_grid_operators(N_pix)
    max_rel_diff_lambda_res = float(
        np.max(np.abs(ops_160["sig_2d_grid_sh"][:4] - ops_128["sig_2d_grid_sh"][:4]) / sig_2d_cont_sh[:4])
    )
    assert max_rel_diff_lambda_res < 0.015, f"Low-z Parseval variance mismatch: {max_rel_diff_lambda_res}"

    L_mag = ops_160["L_mag"]
    cos2psi = ops_160["cos2psi"]
    sin2psi = ops_160["sin2psi"]
    kxhat2_m_13 = ops_160["kxhat2_m_13"]
    kyhat2_m_13 = ops_160["kyhat2_m_13"]
    kxhyh = ops_160["kxhyh"]
    C_target_sh = ops_160["C_target_sh"]
    C_gauss_cj_sh = ops_160["C_gauss_cj_sh"]
    var_gauss_cj_sh = ops_160["var_gauss_cj_sh"]
    W_sm_2d_sh = ops_160["W_sm_2d_sh"]
    lambda_3d_2d_sh = ops_160["lambda_3d_2d_sh"]
    RW_sh_2d = ops_160["RW_2d_sh"]

    ell_bp_edges = np.geomspace(30.0, 1000.0, 9)
    n_bp = ell_bp_edges.size - 1
    bp_masks = [(L_mag >= ell_bp_edges[b]) & (L_mag < ell_bp_edges[b + 1]) for b in range(n_bp)]
    n_modes_bp = np.array([float(m.sum()) for m in bp_masks], dtype=np.float64)
    ell_bp_mid = np.array([float(np.mean(L_mag[m])) for m in bp_masks])

    # Gate 3B.2: Verify Coles-Jones (1991) inverse log-normal mapping reproduces C_target_sh[10] within sample error!
    cj_rng = np.random.default_rng(20261199)
    norm_cell = Omega_patch / (float(N_pix) ** 4)
    sh_idx_chk = 10
    emp_cj_bp = np.zeros(n_bp, dtype=np.float64)
    n_cj_chk = 48
    for _ in range(n_cj_chk):
        fw = np.fft.fft2(cj_rng.normal(0.0, 1.0, size=(N_pix, N_pix)))
        dG = np.fft.ifft2((N_pix / math.sqrt(Omega_patch)) * np.sqrt(C_gauss_cj_sh[sh_idx_chk]) * fw).real
        dLN = np.exp(dG - 0.5 * float(np.var(dG))) - 1.0
        dLN_k = np.fft.fft2(dLN)
        for b_i, msk in enumerate(bp_masks):
            emp_cj_bp[b_i] += norm_cell * float(np.mean(np.abs(dLN_k[msk]) ** 2)) / n_cj_chk
    targ_cj_bp = np.array([float(np.mean(C_target_sh[sh_idx_chk][msk])) for msk in bp_masks])
    # Check intermediate/high-mode bandpowers b >= 2 (M_b >= 60 -> 48 * 60 = 2880 dof, < 8% tolerance):
    max_cj_bp_rel_err = float(np.max(np.abs(emp_cj_bp[2:] - targ_cj_bp[2:]) / targ_cj_bp[2:]))
    assert max_cj_bp_rel_err < 0.08, f"Coles-Jones log-normal power spectrum mismatch: {max_cj_bp_rel_err}"

    def simulate_lightcone_map(
        seed_val: int,
        alpha_true: float,
        add_shape_noise: bool = True,
        gaussian_linear_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a 10-bin Euclid lightcone shear realization at alpha_true in {0.0, 1.0}."""
        loc_rng = np.random.default_rng(seed_val)
        kap_k = np.zeros((nb, N_pix, N_pix), dtype=np.complex128)
        gam1_ia_x = np.zeros((nb, N_pix, N_pix), dtype=np.float64)
        gam2_ia_x = np.zeros((nb, N_pix, N_pix), dtype=np.float64)

        for is_h in range(n_sh):
            dc_s = float(dchi_sh[is_h])
            fft_white = np.fft.fft2(loc_rng.normal(0.0, 1.0, size=(N_pix, N_pix)))
            if gaussian_linear_only:
                delta_sh_x = np.fft.ifft2(
                    (N_pix / math.sqrt(Omega_patch)) * np.sqrt(C_target_sh[is_h]) * fft_white
                ).real
            else:
                vg = float(var_gauss_cj_sh[is_h])
                if vg > 0.0:
                    dG_x = np.fft.ifft2(
                        (N_pix / math.sqrt(Omega_patch)) * np.sqrt(C_gauss_cj_sh[is_h]) * fft_white
                    ).real
                    delta_sh_x = np.exp(dG_x - 0.5 * vg) - 1.0
                    delta_sh_x -= float(np.mean(delta_sh_x))
                else:
                    delta_sh_x = np.zeros((N_pix, N_pix), dtype=np.float64)

            delta_sh_k = np.fft.fft2(delta_sh_x)
            rw_fac = 1.0 + alpha_true * (RW_sh_2d[is_h] - 1.0)
            kap_k += (delta_sh_k * rw_fac)[None, :, :] * (dc_s * WG_sh[:, is_h])[:, None, None]

            # Smooth shell density/tidal field on Lagrangian halo scale k_sm = 1.0 h_fid Mpc^{-1}:
            d_sm_k = delta_sh_k * W_sm_2d_sh[is_h]
            s_xx_2d = np.fft.ifft2(kxhat2_m_13 * d_sm_k).real
            s_yy_2d = np.fft.ifft2(kyhat2_m_13 * d_sm_k).real
            s_xy_2d = np.fft.ifft2(kxhyh * d_sm_k).real

            if gaussian_linear_only:
                e1_ia_sh = float(C1_sh[is_h]) * (s_xx_2d - s_yy_2d)
                e2_ia_sh = 2.0 * float(C1_sh[is_h]) * s_xy_2d
            else:
                # Exact resolution-independent 3D-to-2D Lagrangian variance factor lambda_s = sigma_{3D,IA} / sigma_{2D,sm}:
                lam_s = float(lambda_3d_2d_sh[is_h])
                d_sm_x = np.fft.ifft2(d_sm_k).real
                s_zz_2d = -(s_xx_2d + s_yy_2d)
                s_sq_2d = (s_xx_2d ** 2) + (s_yy_2d ** 2) + (s_zz_2d ** 2) + 2.0 * (s_xy_2d ** 2)
                t_xx_2d = (s_xx_2d ** 2) + (s_xy_2d ** 2) - (1.0 / 3.0) * s_sq_2d
                t_yy_2d = (s_xy_2d ** 2) + (s_yy_2d ** 2) - (1.0 / 3.0) * s_sq_2d
                t_xy_2d = s_xy_2d * (s_xx_2d + s_yy_2d)

                # Linear TA uses unsmoothed/smoothed projected amplitude (1 * s_ab), while quadratic TATT
                # carries one factor of physical 3D field amplitude (lam_s * d_sm_x) * s_ab_2d and lam_s * t_ab_2d!
                e1_ia_sh = (
                    float(C1_sh[is_h]) * (s_xx_2d - s_yy_2d)
                    + float(C1d_sh[is_h]) * (lam_s * d_sm_x) * (s_xx_2d - s_yy_2d)
                    + float(C2_sh[is_h]) * lam_s * (t_xx_2d - t_yy_2d)
                )
                e2_ia_sh = 2.0 * (
                    float(C1_sh[is_h]) * s_xy_2d
                    + float(C1d_sh[is_h]) * (lam_s * d_sm_x) * s_xy_2d
                    + float(C2_sh[is_h]) * lam_s * t_xy_2d
                )

            gam1_ia_x += (dc_s * nz_sh[:, is_h])[:, None, None] * e1_ia_sh[None, :, :]
            gam2_ia_x += (dc_s * nz_sh[:, is_h])[:, None, None] * e2_ia_sh[None, :, :]

        if add_shape_noise:
            sig_pix_noise = math.sqrt(N_noise_euc) / dx_rad
            n1_x = loc_rng.normal(0.0, sig_pix_noise, size=(nb, N_pix, N_pix))
            n2_x = loc_rng.normal(0.0, sig_pix_noise, size=(nb, N_pix, N_pix))
            gam1_ia_n_k = np.fft.fft2(gam1_ia_x + n1_x)
            gam2_ia_n_k = np.fft.fft2(gam2_ia_x + n2_x)
        else:
            gam1_ia_n_k = np.fft.fft2(gam1_ia_x)
            gam2_ia_n_k = np.fft.fft2(gam2_ia_x)

        g1_k = kap_k * cos2psi[None, :, :] + gam1_ia_n_k
        g2_k = kap_k * sin2psi[None, :, :] + gam2_ia_n_k
        E_k = g1_k * cos2psi[None, :, :] + g2_k * sin2psi[None, :, :]
        B_k = -g1_k * sin2psi[None, :, :] + g2_k * cos2psi[None, :, :]

        EE_bp = np.zeros((n_bp, nb, nb), dtype=np.float64)
        for b_idx, msk in enumerate(bp_masks):
            e_m = E_k[:, msk]
            EE_bp[b_idx] = norm_cell * np.einsum("im,jm->ij", e_m, e_m.conj()).real / msk.sum()
        return EE_bp, B_k * math.sqrt(norm_cell)

    # 1. Exact paired E -> B gravitational lensing leakage unit test:
    _, B_pair_tau = simulate_lightcone_map(2026112600, 1.0, add_shape_noise=True)
    _, B_pair_gr = simulate_lightcone_map(2026112600, 0.0, add_shape_noise=True)
    max_bb_leak = float(np.max(np.abs(B_pair_tau - B_pair_gr)))
    assert max_bb_leak < 1.0e-15

    # 2. Clean Noiseless Signal Template + Independent 192-Mock Hartlap-Debiased Baseline Calibration:
    #    - 64 NOISELESS paired realizations (`add_shape_noise=False`, seeds 4000..4063) determine
    #      Delta_C_bp = <EE_tau^{noiseless} - EE_gr^{noiseless}> with ZERO shape-noise regression dilution!
    #    - 192 INDEPENDENT noisy GR calibration realizations (seeds 3000..3191, completely disjoint from
    #      test mocks!) determine BOTH the baseline mean C_0_cal AND the Hartlap-debiased (Hartlap et al.
    #      2007, A&A 464, 399) inverse bandpower matrix C_inv_bp = [(nu_b - p - 1) / nu_b] * C_0_cal^{-1}!
    N_tmpl_noiseless = 64
    tmpl_diffs = [
        simulate_lightcone_map(2026114000 + t_i, 1.0, add_shape_noise=False)[0]
        - simulate_lightcone_map(2026114000 + t_i, 0.0, add_shape_noise=False)[0]
        for t_i in range(N_tmpl_noiseless)
    ]
    Delta_C_bp = np.mean(np.array(tmpl_diffs), axis=0)

    N_cal_gr = 192
    cal_gr_arr = np.array([simulate_lightcone_map(2026113000 + c_i, 0.0, add_shape_noise=True)[0] for c_i in range(N_cal_gr)])
    C_0_cal_bp = np.mean(cal_gr_arr, axis=0)

    W_opt_bp = np.zeros_like(Delta_C_bp)
    F_patch_wishart = 0.0
    for b_idx in range(n_bp):
        nu_b = float(N_cal_gr) * float(n_modes_bp[b_idx])
        hartlap_fac = (nu_b - float(nb) - 1.0) / nu_b
        assert hartlap_fac > 0.90
        ci = hartlap_fac * np.linalg.inv(C_0_cal_bp[b_idx])
        dc = Delta_C_bp[b_idx]
        ci_dc = mm(ci, dc)
        W_opt_bp[b_idx] = 0.5 * float(n_modes_bp[b_idx]) * mm(ci_dc, ci)
        F_patch_wishart += 0.5 * float(n_modes_bp[b_idx]) * float(np.trace(mm(ci_dc, ci_dc)))

    # 3. Evaluate on THREE disjoint 64-mock test ensembles centered on the SAME independent C_0_cal_bp
    #    (ZERO Leave-One-Out tautology!):
    #    (a) 64 independent Tau-Core Coles-Jones + FFT-TATT test mocks (seeds 1000..1063, alpha_true = 1.0);
    #    (b) 64 independent Null-GR Coles-Jones + FFT-TATT test mocks  (seeds 2000..2063, alpha_true = 0.0);
    #    (c) 64 paired Pure-Gaussian linear-IA control mocks           (seeds 2000..2063, gaussian_linear_only=True)
    #        centered on their own Gaussian calibration mean to isolate pure log-normal + nonlinear-TATT
    #        variance inflation cleanly!
    N_mocks = 64
    ee_tau_indep = np.array([simulate_lightcone_map(2026111000 + m_i, 1.0, add_shape_noise=True)[0] for m_i in range(N_mocks)])
    ee_gr_indep = np.array([simulate_lightcone_map(2026112000 + m_i, 0.0, add_shape_noise=True)[0] for m_i in range(N_mocks)])
    ee_gauss_ctrl = np.array(
        [simulate_lightcone_map(2026112000 + m_i, 0.0, add_shape_noise=True, gaussian_linear_only=True)[0] for m_i in range(N_mocks)]
    )

    alpha_hat_tau = np.einsum("bij,mbji->m", W_opt_bp, ee_tau_indep - C_0_cal_bp[None]) / F_patch_wishart
    alpha_hat_gr = np.einsum("bij,mbji->m", W_opt_bp, ee_gr_indep - C_0_cal_bp[None]) / F_patch_wishart
    alpha_hat_gauss = np.einsum("bij,mbji->m", W_opt_bp, ee_gauss_ctrl - np.mean(ee_gauss_ctrl, axis=0)[None]) / F_patch_wishart

    f_sky_patch = Omega_patch / (4.0 * math.pi)
    area_boost = f_sky_euc / f_sky_patch

    mean_tau_hat = float(np.mean(alpha_hat_tau))
    mean_gr_hat = float(np.mean(alpha_hat_gr))
    std_patch_tau = float(np.std(alpha_hat_tau, ddof=1))
    std_patch_gr = float(np.std(alpha_hat_gr, ddof=1))
    std_patch_ln_avg = 0.5 * (std_patch_tau + std_patch_gr)
    std_patch_gauss_ctrl = float(np.std(alpha_hat_gauss, ddof=1))
    ng_inflation_vs_gauss_ctrl = std_patch_ln_avg / std_patch_gauss_ctrl

    # Exact total standard error of (mean_test - mean_cal) including both 1/N_mocks and 1/N_cal_gr:
    se_mean_tot = std_patch_ln_avg * math.sqrt(1.0 / float(N_mocks) + 1.0 / float(N_cal_gr))
    assert abs(mean_tau_hat - 1.0) < 2.5 * se_mean_tot, f"Biased Tau-Core mock recovery: {mean_tau_hat} (se={se_mean_tot})"
    assert abs(mean_gr_hat) < 2.5 * se_mean_tot, f"Biased Null-GR mock recovery: {mean_gr_hat} (se={se_mean_tot})"
    assert abs(mean_gr_hat) > 1.0e-8, "Null-GR mean is suspiciously exact zero (check against tautology!)"

    std_euclid_unmarg_mock = std_patch_ln_avg / math.sqrt(area_boost)
    snr_euclid_unmarg_mock = 1.0 / std_euclid_unmarg_mock

    snr_analytic_1param_1000 = math.sqrt(float(F11_high_1000[0, 0]))
    deg_factor_plk = res_11_high_1000["SNR_11param_FASTPT_full_Euclid_plus_Planck18"] / snr_analytic_1param_1000
    deg_factor_euc = res_11_high_1000["SNR_11param_FASTPT_full_Euclid_alone"] / snr_analytic_1param_1000
    snr_mock_calibrated_11param_plk = snr_euclid_unmarg_mock * deg_factor_plk
    snr_mock_calibrated_11param_euc = snr_euclid_unmarg_mock * deg_factor_euc
    assert snr_mock_calibrated_11param_plk > 4.0

    print(
        f"[Module K Rev2 Summary] 11p-FASTPT(ell<=1500): Euc={res_11_high_1500['SNR_11param_FASTPT_full_Euclid_alone']}s, "
        f"+Plk={res_11_high_1500['SNR_11param_FASTPT_full_Euclid_plus_Planck18']}s | "
        f"11p-FASTPT(ell<=1000): Euc={res_11_high_1000['SNR_11param_FASTPT_full_Euclid_alone']}s, "
        f"+Plk={res_11_high_1000['SNR_11param_FASTPT_full_Euclid_plus_Planck18']}s | "
        f"Lightcone64: <a_tau>={mean_tau_hat:.3f}, <a_gr>={mean_gr_hat:+.3f} (SE={se_mean_tot:.3f}), "
        f"1p_mock={snr_euclid_unmarg_mock:.2f}s (NG_vs_GaussCtrl={ng_inflation_vs_gauss_ctrl:.3f}), "
        f"mock-11p+Plk={snr_mock_calibrated_11param_plk:.2f}s"
    )

    return {
        "gate1b_7param_nla_benchmark_check": {
            "computed_SNR_Euclid_alone": round(snr_7_euc, 3),
            "module_C_reference_Euclid_alone": round(modc_snr_euc, 3),
            "computed_SNR_Euclid_plus_Planck18": round(snr_7_plk, 3),
            "module_C_reference_Euclid_plus_Planck18": round(modc_snr_plk, 3),
        },
        "fastpt_1loop_basis_summary_at_k0p2_Mpc_inv": {
            "I_wn0_white_noise_integral": fpt_fid["I_wn0"],
            "P_0_0E_tot_ren_k0p2": float(np.interp(0.2, k_hf, fpt_fid["P_0_0E_tot_ren"])),
            "P_0_E2_mc_only_k0p2": float(np.interp(0.2, k_hf, fpt_fid["P_0_E2_mc"])),
            "P_0_E2_tot_ren_with_4B_k0p2": float(np.interp(0.2, k_hf, fpt_fid["P_0_E2_tot_ren"])),
            "P_0E_0E_raw_k0p2": float(np.interp(0.2, k_hf, fpt_fid["P_0E_0E_raw"])),
            "P_E2_E2_raw_k0p2": float(np.interp(0.2, k_hf, fpt_fid["P_E2_E2_raw"])),
        },
        "eleven_param_fastpt_names": param_names_11,
        "high_IA_bTA1_TT_AGN_ell_le_1500": res_11_high_1500,
        "high_IA_bTA1_TT_AGN_ell_le_1000_conservative": res_11_high_1000,
        "high_IA_bTA1_TT_AGN_ell_le_1000_ir_highpass_filter_check": res_11_high_1000_irhp,
        "DES_like_IA_bTA1_TT_AGN_ell_le_1000": res_11_des_1000,
        "lightcone_3d_coles_jones_tatt_mocks_64": {
            "n_test_mocks_per_hypothesis": N_mocks,
            "n_noiseless_template_pairs": N_tmpl_noiseless,
            "n_independent_gr_calibration_mocks": N_cal_gr,
            "n_radial_shells": n_sh,
            "grid_pixels": f"{N_pix}x{N_pix}",
            "patch_area_deg2": L_deg ** 2,
            "euclid_area_boost_factor": round(area_boost, 3),
            "ell_bandpower_centers": [round(v, 1) for v in ell_bp_mid],
            "sigma_3D_IA_Lagrangian_z0": round(sig_3d_ia_z0, 4),
            "max_rel_diff_lambda_Npix128_vs_160": round(max_rel_diff_lambda_res, 6),
            "coles_jones_shell10_power_max_rel_err": round(max_cj_bp_rel_err, 5),
            "max_spurious_EB_lensing_leakage": max_bb_leak,
            "mean_alpha_hat_tau_core_mocks": round(mean_tau_hat, 5),
            "mean_alpha_hat_null_GR_mocks": round(mean_gr_hat, 5),
            "se_sample_mean_including_calibration": round(se_mean_tot, 5),
            "std_alpha_hat_single_patch_LN_TATT": round(std_patch_ln_avg, 4),
            "std_alpha_hat_single_patch_Gaussian_control": round(std_patch_gauss_ctrl, 4),
            "non_gaussian_variance_inflation_vs_gaussian_control": round(ng_inflation_vs_gauss_ctrl, 4),
            "SNR_unmarginalized_lightcone_mocks_full_Euclid": round(snr_euclid_unmarg_mock, 3),
            "SNR_unmarginalized_analytic_Fisher_ell_le_1000": round(snr_analytic_1param_1000, 3),
            "SNR_mock_calibrated_11param_Euclid_alone": round(snr_mock_calibrated_11param_euc, 3),
            "SNR_mock_calibrated_11param_Euclid_plus_Planck18": round(snr_mock_calibrated_11param_plk, 3),
        },
    }


def main() -> None:
    res_k12 = gate1a_and_2a_24cell_microscopic_dc_derivation()
    sym_tatt = derive_sympy_fastpt_tatt_and_spt_kernels()
    res_fastpt_lc = gate2bc_fastpt_11param_euclid_and_3d_lightcone_mocks(sym_tatt)

    cand_a = res_k12["conditional_candidate_A_12link_heat_kernel_matched"]
    aM_str = f"{cand_a['a_times_M_star']:.6f}"
    deb_str = f"{cand_a['Debye_Lambda_BZ_over_M_star']:.4f}"

    out = {
        "module": "Paper 29 v3.6.0 — Module K (Rev 2): Microscopic 4D 24-Cell (D_4) Lattice Audit of (d_max, d_c), Complete 1-Loop FAST-PT 11-Parameter TATT Forecast & Coles-Jones Lightcone Mocks",
        "NATURE_LEVEL_CLOSURE": False,
        "epistemic_tiers": {
            "PROVEN": [
                "Luscher-Weisz / Capitani (2003) 4D Z^4 massless tadpole Z_0(Z^4) = 0.15493339023106021 reproduced to < 10^{-15}, and Hasenfratz-Karsch (1983) O(a^{-2}) vacuum UV-divergence cancellation proven on both Z^4 and 24-cell (D_4).",
                "Discrete K_4 subset 24-cell transverse SO(2) spatial director quadratic Casimir d_max = (1/2) Tr(Pi_perp T_def^2) == u^2 = 1/192, 12-link seed identity 12 u^2 == r_seed^2 = 1/16, Hasenfratz-Karsch link-reversal parity generating 2 K_2 Q_0^2 d_eff^2 [cosh(Delta_Q / d_eff) - 1], and exact Voronoi V_0 = 1/2 continuum tadpole Jacobian G_{D_4}(0; a) == 2 Z_0(D_4) / a^2.",
                "Exact 4D 24-cell (D_4) Watson integral Z_0(D_4) = 0.138230474737 cross-verified by Richardson BZ extrapolation and 1D Hubbard-Stratonovich proper-time integration (< 10^{-9}), with W(F_4)-transitive per-link resolvent Pi_{mm}^{(D_4)} == (1/4) Z_0(D_4).",
                f"Algebraic Watson-Constant Cancellation Identity: if hat{{D}} couples to 12 independent canonical link modes and the lattice spacing is matched via G_lat(0; a) = M_*^2 / (16 pi^2) (a M_* = {aM_str}, Debye Lambda_{{BZ}} = {deb_str} M_* on D_4), Z_0 cancels algebraically to yield d_c = (3 / (4 pi^2)) u^4 = (m_R / M_*) d_max.",
                "First-principles SymPy derivation of ALL Blazek et al. (2019) 1-loop FAST-PT TATT mode-coupling AND 1PR propagator kernels (including both B_{0|0E} and 4 B_{0|E2}) + Makino-Sasaki-Suto (1992) P_{22} + P_{13} == 55/784 benchmark, proving exact Galilean IR pole cancellations (5816/2205, 251/105, 32/315) and k -> 0 E/B parity.",
            ],
            "PROVEN_NO_GO": [
                "Naive unmatched bare edge identification a = l_* = M_*^{-1} (a M_* = 1) on the 24-cell lattice violates the Bridge-2D Cosmic-Jeans upper bound d_c <= 2.41e-6 for BOTH the 12-link sum (d_c = 24 Z_0(D_4) u^4 = 8.9994e-5, 37.34x violation) AND even the single W(F_4)-invariant radial-breathing singlet (d_c = 2 Z_0(D_4) u^4 = 7.4995e-6, 3.11x violation; and 2.4998e-6 at a M_* = sqrt(3)).",
                "Naive linear chemical-potential lattice coupling (omitting Hasenfratz-Karsch link exponentials) generates a spurious O(a^{-2}) vacuum susceptibility UV divergence on both Z^4 and D_4.",
            ],
            "CONDITIONAL": [
                f"Candidate A (12-link quadratic strain + heat-kernel tadpole matching G_{{D_4}}(0; a) = M_*^2 / (16 pi^2), a M_* = {aM_str}): yields d_c = (3 / (4 pi^2)) u^4 = 2.061385e-6; Candidate B (1-mode W(F_4) radial-breathing singlet at 24-cell circumdiameter scale a = 2 l_* = 2 M_*^{{-1}}, inside viable window a M_* in [1.7641, 2.2359]): yields d_c = (1/2) Z_0(D_4) u^4 = 1.874871e-6 (verified across all 4,001 radial shells of Module F's 3-field Schur BVP), alongside auxiliary threshold sig_c = 10^{{-13}}.",
                "Simultaneous 11-parameter Stage-IV Euclid nonlocal FAST-PT TATT (including b_TA and complete 4 B_{0|E2}) + AGN Fisher forecast and 64-realization 24-shell Coles-Jones log-normal + resolution-independent FFT-TATT lightcone shear mock validation.",
            ],
            "OPEN": [
                "Non-perturbative 24-cell Regge / spin-foam derivation fixing the UV scale matching ratio a M_* and composite operator hat{D}[sigma_scr, hat{xi}] (and sig_c) without continuum heat-kernel or circumdiameter matching, plus N-body hydrodynamical Euclid Flagship lightcone mocks and external human peer review.",
            ],
        },
        "gate1a_2a_24cell_microscopic_dc": res_k12,
        "gate1b_2b_sympy_fastpt_benchmarks": sym_tatt["benchmarks"],
        "gate2bc_11param_fastpt_and_lightcone_mocks": res_fastpt_lc,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "24cell_microscopic_dc_and_fastpt_lightcone.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("MODULE_K_REV2_24CELL_MICROSCOPIC_DC_AND_FASTPT_LIGHTCONE_PASS")
    print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    main()
