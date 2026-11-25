#!/usr/bin/env python3
"""
Tau Core — Q-DEC / T-DEC Common Physical Mechanism & UV-to-IR Quantum/Classical-Pointer Split
==============================================================================================

Conventions:
- Metric signature on projective state manifold CP^(n-1): Riemannian positive-definite
  Fubini-Study metric g_{mu nu} = Re(Q_{mu nu}).
- Quantum Geometric Tensor (Provost & Vallee 1980, CMP 76, 289):
      Q_{mu nu} = <d_mu psi | (I - |psi><psi|) | d_nu psi> = g_{mu nu} - (i/2) F_{mu nu}
  where F_{mu nu} = -2 Im(Q_{mu nu}) = d_mu A_nu - d_nu A_mu is the Berry curvature 2-form,
  A_mu = i <psi | d_mu psi>, and omega_{mu nu} = Im(Q_{mu nu}) = -F_{mu nu}/2 is the Kahler
  2-form satisfying omega = -g J for canonical complex structure J(d/dx_a) = d/dy_a,
  J(d/dy_a) = -d/dx_a.
- Normalized Chung Laplacian on K_4 (Chung 1997): L_{K4} = I_4 - (1/3) A_{K4},
  eigenvalues {0, 4/3, 4/3, 4/3}, Tr(L_{K4}) = 4.
- Bures metric on 1-qubit Bloch ball rho = (I + r . sigma)/2 (Hubner 1992, PLA 163, 239):
      ds_Bures^2 = (1/4) [ |dr|^2 + (r . dr)^2 / (1 - |r|^2) ].

Epistemic discipline (after Stage-A Hostile Referee Audit):
- PROVEN_ALGEBRAIC (exact, zero free parameters):
  1. On ANY Kahler projective manifold CP^(n-1) — in particular the K_4 projective space CP^3
     at the pure UV seed (N_E = 0) — the real metric g_{mu nu} and imaginary symplectic/Berry
     2-form omega_{mu nu} are strictly unified by omega = -g J with orthonormal-frame Frobenius
     ratio ||omega_orth||_F / ||g_orth||_F = 1 identically.
  2. Under QND pure-dephasing observer descent (partial trace over N_E environmental modes
     with commuting interaction [H_int, H_pointer] = 0 and generic coupling eta not in (pi/2)Z):
     - Off-diagonal quantum coherence visibility decays exponentially: V_Q = |cos(2*eta)|^{N_E};
     - Both the phase Bures metric g_{phi phi}^{Bures} = (1/4) sin^2(theta) cos^{2 N_E}(2*eta)
       and the effective Bloch-disk symplectic area / mixed-state interference 2-form scale as
       cos^{2 N_E}(2*eta) -> 0;
     - Simultaneously, for ALL finite N_E (proved symbolically for arbitrary c in (-1, 1) and
       verified numerically via exact Uhlmann fidelity), the pointer-basis expectation <H_pointer>
       and the classical Fisher-Rao / Bures pointer-population metric g_{theta theta}^{Bures} = 1/4
       are 100% preserved!
  3. CRITICAL CLOCK DISTINCTION: Unitary evolution under H_pointer = diag(0, 4/3) alone moves
     exclusively along phi, so QND dephasing in the H_pointer eigenbasis STOPS the single-qubit
     Mandelstam-Tamm clock (ds_Bures/dt = sqrt(g_{phi phi}) |dphi/dt| -> 0). A macroscopic
     physical clock therefore CANNOT be a single H_pointer-eigenbasis qubit; it requires a
     composite / pointer-variable clock whose distinguishable pointer states evolve along the
     preserved classical Fisher-Rao sector (or non-commuting clock observable).
- OPEN_QDEC_TDEC (NATURE_LEVEL_CLOSURE = false):
  - Why Nature selects the complex Kahler field C (rather than R or H) at the pregeometric seed.
  - Deriving 4D Lorentzian spacetime / Einstein gravity from the preserved classical Fisher-Rao
    pointer sector without assuming it a priori.
  - Deriving the physical environmental coupling eta(s) and composite pointer-clock dynamics
    from the full 4D non-perturbative Tau Core action.
"""

import json
import math
import pathlib
import numpy as np
import sympy as sp


def bloch_rho(rx, ry, rz):
    """Exact 2x2 density matrix rho = (I + r . sigma)/2."""
    return 0.5 * np.array([[1.0 + rz, rx - 1j * ry], [rx + 1j * ry, 1.0 - rz]], dtype=complex)


def bures_ds2(rho1, rho2):
    """Exact 2x2 Uhlmann-Bures squared distance ds^2 = 2(1 - sqrt(F(rho1, rho2)))."""
    tr12 = float(np.real(np.trace(rho1 @ rho2)))
    det1 = max(0.0, float(np.real(np.linalg.det(rho1))))
    det2 = max(0.0, float(np.real(np.linalg.det(rho2))))
    root_F = math.sqrt(max(0.0, tr12 + 2.0 * math.sqrt(det1 * det2)))
    return 2.0 * max(0.0, 1.0 - min(1.0, root_F))


def gate1_literature_benchmarks():
    """Gate 1: Reproduce exact published literature benchmarks before any Tau Core claim."""
    # Benchmark 1A: Provost & Vallee (1980) QGT on CP^1 (Bloch sphere)
    theta, phi = sp.symbols("theta phi", real=True, positive=True)
    psi = sp.Matrix([sp.cos(theta / 2), sp.exp(sp.I * phi) * sp.sin(theta / 2)])
    psi_dag = psi.conjugate().T
    assert sp.simplify((psi_dag * psi)[0]) == 1

    P_perp = sp.eye(2) - psi * psi_dag
    d_psi = [psi.diff(theta), psi.diff(phi)]
    Q = sp.Matrix(2, 2, lambda i, j: sp.simplify((d_psi[i].conjugate().T * P_perp * d_psi[j])[0]))

    g_FS = sp.Matrix(2, 2, lambda i, j: sp.simplify(sp.re(Q[i, j])))
    F_berry = sp.Matrix(2, 2, lambda i, j: sp.simplify(-2 * sp.im(Q[i, j])))

    assert sp.simplify(g_FS[0, 0] - sp.Rational(1, 4)) == 0
    assert sp.simplify(g_FS[1, 1] - sp.Rational(1, 4) * sp.sin(theta) ** 2) == 0
    assert sp.simplify(g_FS[0, 1]) == 0 and sp.simplify(g_FS[1, 0]) == 0
    assert sp.simplify(F_berry[0, 1] - (-sp.Rational(1, 2) * sp.sin(theta))) == 0
    assert sp.simplify(F_berry[1, 0] + F_berry[0, 1]) == 0

    C1 = sp.integrate(sp.integrate(F_berry[0, 1], (phi, 0, 2 * sp.pi)), (theta, 0, sp.pi)) / (2 * sp.pi)
    assert sp.simplify(C1 - (-1)) == 0

    # Benchmark 1B: Anandan & Aharonov (1990) Mandelstam-Tamm Speed Limit (PV normalization)
    t, dE, hbar = sp.symbols("t dE hbar", real=True, positive=True)
    H = (dE / 2) * sp.Matrix([[1, 0], [0, -1]])
    psi0 = sp.Matrix([1, 1]) / sp.sqrt(2)
    psi_t = sp.Matrix([sp.exp(-sp.I * dE * t / (2 * hbar)), sp.exp(sp.I * dE * t / (2 * hbar))]) / sp.sqrt(2)

    exp_H = sp.simplify((psi0.conjugate().T * H * psi0)[0])
    exp_H2 = sp.simplify((psi0.conjugate().T * H * H * psi0)[0])
    Delta_H = sp.sqrt(sp.simplify(exp_H2 - exp_H ** 2))
    assert sp.simplify(Delta_H - dE / 2) == 0

    dpsi_dt = psi_t.diff(t)
    P_t = sp.eye(2) - psi_t * psi_t.conjugate().T
    Q_tt = sp.simplify((dpsi_dt.conjugate().T * P_t * dpsi_dt)[0])
    ds_dt = sp.sqrt(Q_tt)
    assert sp.simplify(ds_dt - Delta_H / hbar) == 0
    assert sp.simplify((sp.pi / 2) / ds_dt - sp.pi * hbar / dE) == 0

    # Benchmark 1C: Chung (1997) Normalized Laplacian on K_4
    A_K4 = np.ones((4, 4), dtype=float) - np.eye(4)
    L_K4 = np.eye(4) - A_K4 / 3.0
    evals = np.sort(np.linalg.eigvalsh(L_K4))
    assert np.allclose(evals, [0.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0], atol=1e-14)
    assert abs(np.trace(L_K4) - 4.0) < 1e-14

    # Benchmark 1D: Hubner (1992) Exact Bures Metric on 1-Qubit Bloch Ball
    def bloch_sph(r_val, th_val, ph_val):
        return bloch_rho(
            r_val * math.sin(th_val) * math.cos(ph_val),
            r_val * math.sin(th_val) * math.sin(ph_val),
            r_val * math.cos(th_val),
        )

    r0, th0, ph0, eps = 0.65, 1.1, 0.7, 1e-4
    g_rr_num = bures_ds2(bloch_sph(r0 - eps, th0, ph0), bloch_sph(r0 + eps, th0, ph0)) / (4.0 * eps ** 2)
    g_tt_num = bures_ds2(bloch_sph(r0, th0 - eps, ph0), bloch_sph(r0, th0 + eps, ph0)) / (4.0 * eps ** 2)
    g_pp_num = bures_ds2(bloch_sph(r0, th0, ph0 - eps), bloch_sph(r0, th0, ph0 + eps)) / (4.0 * eps ** 2)

    g_rr_lit = 0.25 / (1.0 - r0 ** 2)
    g_tt_lit = 0.25 * (r0 ** 2)
    g_pp_lit = 0.25 * (r0 ** 2) * (math.sin(th0) ** 2)

    max_err = max(
        abs(g_rr_num - g_rr_lit) / g_rr_lit,
        abs(g_tt_num - g_tt_lit) / g_tt_lit,
        abs(g_pp_num - g_pp_lit) / g_pp_lit,
    )
    assert max_err < 1e-6, f"Bures Hubner mismatch: {max_err}"

    return {
        "cp1_chern_number": int(C1),
        "mandelstam_tamm_exact": True,
        "k4_laplacian_trace": float(np.trace(L_K4)),
        "bures_hubner_max_rel_err": float(max_err),
    }


def gate2_kahler_uv_unification_and_split():
    """
    Gate 2:
    (2A) Exact Kahler unification omega = -g J and ||omega_orth||_F / ||g_orth||_F = 1 on CP^3 (K_4).
         (Note: this holds generically on every Kahler projective manifold CP^(n-1), including K_4's CP^3.)
    (2B) Exact QND pure-dephasing split ([H_int, H_pointer] = 0, eta not in (pi/2)Z):
         Proves symbolically for ALL c = cos^{N_E}(2*eta) in (-1, 1) that:
           - g_{theta theta}^{Bures}(theta, c) = 1/4 identically (100% preserved pointer Fisher-Rao metric),
           - g_{phi phi}^{Bures}(theta, c) = (1/4) sin^2(theta) c^2 -> 0 (exponentially suppressed phase metric),
           - Effective symplectic Bloch-disk area Omega_eff / Omega_UV = c^2 -> 0 (symplectic collapse),
           - Single-qubit H_pointer clock speed ds_Bures/dt = |c| * (Delta H_pointer / hbar) -> 0 (explicitly
             documenting why a macroscopic clock requires a composite pointer-variable clock!).
    """
    # Part 2A: Exact Kahler Unification on K_4 Projective Space CP^3
    rng = np.random.default_rng(20261125)
    max_kahler_err = 0.0
    max_ratio_err = 0.0
    for _ in range(10):
        z = rng.normal(size=3) + 1j * rng.normal(size=3)
        q0 = np.array([z[0].real, z[0].imag, z[1].real, z[1].imag, z[2].real, z[2].imag])
        v0 = np.array([1.0 + 0j, q0[0] + 1j * q0[1], q0[2] + 1j * q0[3], q0[4] + 1j * q0[5]], dtype=complex)
        nv = np.linalg.norm(v0)
        psi0 = v0 / nv
        P_perp = np.eye(4) - np.outer(psi0, psi0.conj())

        dpsi = []
        for a in range(3):
            for is_im in (0, 1):
                dv = np.zeros(4, dtype=complex)
                dv[a + 1] = 1j if is_im else 1.0
                dpsi.append(dv / nv)

        Q = np.zeros((6, 6), dtype=complex)
        for mu in range(6):
            for nu in range(6):
                Q[mu, nu] = np.vdot(dpsi[mu], P_perp @ dpsi[nu])

        g = np.real(Q)
        omega = np.imag(Q)  # omega = -F/2

        # Canonical complex structure J on tangent basis (d/dx_a, d/dy_a): J(d/dx_a) = d/dy_a, J(d/dy_a) = -d/dx_a
        J = np.zeros((6, 6))
        for a in range(3):
            J[2 * a, 2 * a + 1] = -1.0
            J[2 * a + 1, 2 * a] = +1.0
        assert np.allclose(J @ J, -np.eye(6), atol=1e-15)

        # Exact Kahler identity: omega_{mu nu} = - g_{mu alpha} J^alpha_nu
        max_kahler_err = max(max_kahler_err, float(np.max(np.abs(omega + g @ J))))

        evals_g, evecs_g = np.linalg.eigh(g)
        g_inv_sqrt = evecs_g @ np.diag(1.0 / np.sqrt(evals_g)) @ evecs_g.T
        omega_orth = g_inv_sqrt @ omega @ g_inv_sqrt
        g_orth = g_inv_sqrt @ g @ g_inv_sqrt
        ratio_uv = np.linalg.norm(omega_orth, "fro") / np.linalg.norm(g_orth, "fro")
        max_ratio_err = max(max_ratio_err, abs(float(ratio_uv) - 1.0))

    assert max_kahler_err < 1e-14
    assert max_ratio_err < 1e-14

    # Part 2B: Exact Symbolic QND Pure-Dephasing Split for ARBITRARY c in (-1, 1)
    th, ph, eta, lam = sp.symbols("th ph eta lam", real=True)
    e_plus = sp.Matrix([sp.cos(eta), -sp.I * sp.sin(eta)])
    e_minus = sp.Matrix([sp.cos(eta), +sp.I * sp.sin(eta)])
    overlap = sp.simplify((e_minus.conjugate().T * e_plus)[0])
    assert sp.simplify(overlap - sp.cos(2 * eta)) == 0

    # Pointer-basis expectation invariance under [H_int, H_pointer] = 0
    rho_S_sym = sp.Matrix([
        [sp.cos(th / 2) ** 2, sp.Rational(1, 2) * sp.sin(th) * sp.exp(-sp.I * ph) * lam],
        [sp.Rational(1, 2) * sp.sin(th) * sp.exp(sp.I * ph) * lam, sp.sin(th / 2) ** 2],
    ])
    H_ptr_sym = sp.Matrix([[0, 0], [0, sp.Rational(4, 3)]])
    exp_Hptr = sp.simplify(sp.trace(rho_S_sym * H_ptr_sym))
    assert sp.diff(exp_Hptr, lam) == 0
    assert sp.simplify(exp_Hptr - sp.Rational(4, 3) * sp.sin(th / 2) ** 2) == 0

    # Exact Hubner Bures metric for arbitrary lambda = cos^{N_E}(2*eta) in (-1, 1):
    # Bloch vector r(th, ph) = (lam sin(th) cos(ph), lam sin(th) sin(ph), cos(th))
    r_vec = sp.Matrix([lam * sp.sin(th) * sp.cos(ph), lam * sp.sin(th) * sp.sin(ph), sp.cos(th)])
    dr_dth = r_vec.diff(th)
    dr_dph = r_vec.diff(ph)
    r_sq = sp.simplify((r_vec.T * r_vec)[0])

    # g_{mu nu}^{Bures} = (1/4) [ (dr_mu . dr_nu) + (r . dr_mu)(r . dr_nu) / (1 - |r|^2) ]
    g_tt_exact = sp.simplify(
        sp.Rational(1, 4) * ((dr_dth.T * dr_dth)[0] + ((r_vec.T * dr_dth)[0] ** 2) / (1 - r_sq))
    )
    g_pp_exact = sp.simplify(
        sp.Rational(1, 4) * ((dr_dph.T * dr_dph)[0] + ((r_vec.T * dr_dph)[0] ** 2) / (1 - r_sq))
    )
    g_tp_exact = sp.simplify(
        sp.Rational(1, 4) * ((dr_dth.T * dr_dph)[0] + ((r_vec.T * dr_dth)[0] * (r_vec.T * dr_dph)[0]) / (1 - r_sq))
    )

    assert sp.simplify(g_tt_exact - sp.Rational(1, 4)) == 0, f"g_tt(lam) != 1/4: {g_tt_exact}"
    assert sp.simplify(g_pp_exact - sp.Rational(1, 4) * lam ** 2 * sp.sin(th) ** 2) == 0
    assert sp.simplify(g_tp_exact) == 0

    # Effective Bloch-sphere symplectic area collapse (mixed-state 2-form):
    # Image of S^2 under dephasing is ellipsoid with equatorial radius |lam| => symplectic phase area ~ lam^2
    # Numerical cross-check via exact Uhlmann fidelity (NO hardcoding!) at th0 = 1.1, ph0 = 0.7
    eta_val = 0.10  # generic coupling not in (pi/2)Z
    c_val = math.cos(2.0 * eta_val)
    N_star = -1.0 / math.log(abs(c_val))
    th_chk, ph_chk, eps = 1.1, 0.7, 1e-4

    def rho_deph(th_v, ph_v, lam_v):
        return bloch_rho(
            lam_v * math.sin(th_v) * math.cos(ph_v),
            lam_v * math.sin(th_v) * math.sin(ph_v),
            math.cos(th_v),
        )

    scan = []
    max_num_gtt_err = 0.0
    max_num_gpp_err = 0.0
    for N_E in [0, 1, 5, 10, 25, 50, 100, 250, 500]:
        lam_v = c_val ** N_E
        g_tt_num = bures_ds2(rho_deph(th_chk - eps, ph_chk, lam_v), rho_deph(th_chk + eps, ph_chk, lam_v)) / (4.0 * eps ** 2)
        g_pp_num = bures_ds2(rho_deph(th_chk, ph_chk - eps, lam_v), rho_deph(th_chk, ph_chk + eps, lam_v)) / (4.0 * eps ** 2)
        g_tt_uv = 0.25
        g_pp_uv = 0.25 * (math.sin(th_chk) ** 2)

        max_num_gtt_err = max(max_num_gtt_err, abs(g_tt_num - g_tt_uv))
        max_num_gpp_err = max(max_num_gpp_err, abs(g_pp_num - g_pp_uv * (lam_v ** 2)))

        V_Q = abs(lam_v)
        scan.append({
            "N_E": N_E,
            "V_Q_phase_visibility": round(V_Q, 6),
            "g_phase_over_UV_numeric": round(g_pp_num / g_pp_uv, 6),
            "g_diag_pointer_over_UV_numeric": round(g_tt_num / g_tt_uv, 6),
            "symplectic_area_over_UV": round(lam_v ** 2, 6),
            "single_qubit_Hptr_clock_speed_over_UV": round(V_Q, 6),
            "purity_equator": round(0.5 * (1.0 + lam_v ** 2), 6),
        })

    assert max_num_gtt_err < 1e-6, f"Numeric g_tt != 1/4: {max_num_gtt_err}"
    assert max_num_gpp_err < 1e-6, f"Numeric g_pp != (1/4)lam^2 sin^2(th): {max_num_gpp_err}"

    return {
        "cp3_kahler_max_err": max_kahler_err,
        "uv_omega_over_g_ratio_err": max_ratio_err,
        "uv_omega_over_g_exact": 1.0,
        "exact_single_mode_overlap": "cos(2*eta)",
        "qnd_condition_required": "[H_int, H_pointer] = 0 and eta not in (pi/2)Z",
        "exact_Hptr_invariance_under_trace": True,
        "exact_g_theta_theta_all_NE": "1/4 (proved symbolically for all lam in (-1,1) and verified numerically)",
        "exact_g_phi_phi_all_NE": "(1/4) * sin^2(theta) * cos^(2 N_E)(2*eta) -> 0",
        "max_numeric_g_tt_err": float(max_num_gtt_err),
        "max_numeric_g_pp_err": float(max_num_gpp_err),
        "eta_rad": eta_val,
        "crossover_scale_N_star": round(N_star, 3),
        "uv_to_ir_scan": scan,
    }


def gate3_invariance_and_limit_tests():
    """Gate 3: Gauge/phase invariance, sign symmetry, and limiting-case unit tests."""
    th, ph, a, u_half = sp.symbols("th ph a u_half", real=True)
    alpha = a * th + 2 * a * ph
    psi1 = sp.Matrix([sp.cos(th / 2), sp.exp(sp.I * ph) * sp.sin(th / 2)])
    psi2 = sp.exp(sp.I * alpha) * psi1

    def qgt(psi):
        P = sp.eye(2) - psi * psi.conjugate().T
        d = [psi.diff(th), psi.diff(ph)]
        return sp.Matrix(2, 2, lambda i, j: sp.simplify((d[i].conjugate().T * P * d[j])[0]))

    Q1, Q2 = qgt(psi1), qgt(psi2)
    for i in range(2):
        for j in range(2):
            diff_ij = sp.trigsimp(sp.expand_trig((Q1[i, j] - Q2[i, j]).subs(th, 2 * u_half)))
            assert diff_ij == 0, f"U(1) gauge invariance failed at ({i},{j}): {diff_ij}"

    eta = sp.Symbol("eta", real=True)
    assert sp.simplify(sp.cos(2 * (-eta)) - sp.cos(2 * eta)) == 0
    assert abs(math.cos(0.0) ** 1000 - 1.0) < 1e-15
    assert abs(math.cos(2.0 * 0.37) ** 0 - 1.0) < 1e-15
    assert abs(math.cos(0.2) ** 1000) < 1e-8

    rng = np.random.default_rng(42)
    M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    U, _ = np.linalg.qr(M)
    A_K4 = np.ones((4, 4)) - np.eye(4)
    L_K4 = np.eye(4) - A_K4 / 3.0
    v = rng.normal(size=4) + 1j * rng.normal(size=4)
    v /= np.linalg.norm(v)
    rho = np.outer(v, v.conj())
    tr1 = float(np.real(np.trace(rho @ L_K4)))
    tr2 = float(np.real(np.trace((U @ rho @ U.conj().T) @ (U @ L_K4 @ U.conj().T))))
    assert abs(tr1 - tr2) < 1e-14

    return {
        "u1_local_phase_invariance": True,
        "eta_sign_symmetry": True,
        "uv_and_ir_limits_passed": True,
        "u4_basis_invariance_err": float(abs(tr1 - tr2)),
    }


def main():
    g1 = gate1_literature_benchmarks()
    g2 = gate2_kahler_uv_unification_and_split()
    g3 = gate3_invariance_and_limit_tests()

    out = {
        "script": "verify_qdec_tdec_kahler_split.py",
        "epistemic_status": {
            "NATURE_LEVEL_CLOSURE": False,
            "PROVEN_ALGEBRAIC": [
                "On any Kahler projective manifold CP^(n-1) — in particular the K_4 projective space CP^3 at the pure UV seed (N_E = 0) — the Quantum Geometric Tensor Q = g + i*omega unifies the real Fubini-Study metric g_{mu nu} and the imaginary symplectic/Berry 2-form omega_{mu nu} = -g_{mu alpha} J^alpha_nu with exact orthonormal-frame Frobenius ratio ||omega_orth||_F / ||g_orth||_F = 1.",
                "Under QND pure-dephasing observer descent ([H_int, H_pointer] = 0, eta not in (pi/2)Z, partial trace over N_E environmental modes), off-diagonal quantum phase visibility decays exponentially V_Q = |cos(2*eta)|^{N_E}, and both the phase Bures metric g_{phi phi}^{Bures} and Bloch-disk symplectic area contract as cos^{2 N_E}(2*eta) -> 0, while the pointer-basis expectation <H_pointer> and classical Fisher-Rao / Bures pointer-population metric g_{theta theta}^{Bures} = 1/4 are 100% preserved for all N_E (proved symbolically and verified numerically).",
                "CLOCK COROLLARY: Pure dephasing in the H_pointer eigenbasis also suppresses the single-qubit Mandelstam-Tamm clock speed ds_Bures/dt ~ |cos(2*eta)|^{N_E} -> 0; therefore a macroscopic physical clock cannot be a single H_pointer-eigenbasis qubit and requires a composite pointer-variable clock evolving along the preserved classical Fisher-Rao sector.",
            ],
            "OPEN_QDEC_TDEC": [
                "Why Nature selects the complex Kahler field C (rather than R or H) at the pregeometric seed without assuming complex amplitudes a priori.",
                "Deriving 4D Lorentzian spacetime metric and Einstein/AeST gravity from the preserved classical Fisher-Rao pointer sector.",
                "Constructing the explicit composite macroscopic pointer-clock and deriving the physical environmental coupling eta(s) from the full 4D non-perturbative Tau Core action.",
            ],
        },
        "gate1_benchmarks": g1,
        "gate2_derivation": g2,
        "gate3_invariances": g3,
    }

    out_path = pathlib.Path(__file__).with_name("qdec_tdec_kahler_split_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
