#!/usr/bin/env python3
"""
Tau Core — Q-DEC Field Selection (R vs C vs H on K_4) & T-DEC Composite Pointer-Clock Survival
==============================================================================================

Conventions:
- Division algebras over R (Frobenius theorem): F in {R, C, H} with real dimension beta in {1, 2, 4}.
- Real dimension of trace-1 Hermitian density matrices on F^d (Hardy 2001, quant-ph/0101012;
  Wootters 1990):
      K_F(d) = d + beta * d*(d - 1)/2 - 1  =>  K_F(d) + 1 = d + beta * d*(d - 1)/2.
  Note on H: Because H is non-commutative, H^{d_A} tensor_H H^{d_B} does not exist as a standard
  quaternionic Hilbert space; the naive composite dimension Herm(d_A d_B, H) yields Delta_H < 0
  (over-determination by local observables, consistent with Araki 1980 / Baez 2012).
- GKSL master equation for 2-level coherent clock H = (Omega/2) sigma_x with pointer-basis
  dephasing L = sqrt(gamma/2) sigma_z (hbar = 1):
      drho/dt = -i [H, rho] + (gamma/2) (sigma_z rho sigma_z - rho).
  Exact Bloch equations: dx/dt = -gamma x, dy/dt = -Omega z - gamma y, dz/dt = Omega y.
  Exact eigenvalues in the (y, z) plane: lambda_pm = -gamma/2 +/- sqrt((gamma/2)^2 - Omega^2).
- Classical Fisher-Rao metric on probability simplex {p_m} (matching 1/4 Bures normalization):
      (ds_FR/dt)^2 = (1/4) sum_m (dp_m/dt)^2 / p_m.
- Barato & Seifert (2015, PRL 114, 158101) TUR for steady-state tick count N(T) in a homogeneous
  unicyclic M-state clock with uniform rates Gamma_+ > Gamma_- (Skellam process):
      (Var(N) / <N>^2) * Sigma_T = ((Gamma_+ + Gamma_-) / (Gamma_+ - Gamma_-)) * ln(Gamma_+ / Gamma_-) >= 2.

Epistemic discipline:
- PROVEN_ALGEBRAIC:
  1. Among {R, C, H}, local tomography K_F(d_A d_B) + 1 = (K_F(d_A) + 1)(K_F(d_B) + 1) for all
     d_A, d_B >= 2 (specifically K_4 x K_4: Delta_R = +36, Delta_C = 0, Delta_H = -288) holds
     IF AND ONLY IF beta = 2 (F = C). Moreover, on RP^(d-1) the canonical QGT imaginary part
     Im(Q_{mu nu}) vanishes identically (and dim_R(RP^3) = 3 is odd, admitting no symplectic 2-form).
  2. A coherent single-qubit clock H = (Omega/2) sigma_x under strong pointer-basis dephasing
     gamma >> 2*Omega suffers exact Quantum Zeno suppression Gamma_Zeno = Omega^2 / gamma -> 0.
  3. A composite M-state cyclic pointer-variable clock (diagonal in the decoherence-free pointer
     basis {|m>}) is 100% invariant under arbitrary pointer-basis QND dephasing gamma_dec in [0, inf).
     In steady state (uniform ring marginal p_m^{ss} = 1/M, where periodic marginal ds_FR/dt = 0 for
     all A), the net tick velocity v_tick,ss = Gamma_+ - Gamma_- and entropy production rate
     sigma_dot,ss = (Gamma_+ - Gamma_-) A are strictly positive iff A = ln(Gamma_+/Gamma_-) != 0
     (obeying Barato-Seifert TUR >= 2) and vanish identically iff A = 0; for a localized transient
     pointer hand p0, A = 0 gives symmetric diffusive relaxation while A > 0 adds directed drift
     to both transient Fisher-Rao speed and entropy production.
- OPEN_QDEC_TDEC (NATURE_LEVEL_CLOSURE = false):
  - Why Nature enforces compositional local tomography at the pregeometric K_4 seed.
  - Deriving 4D Lorentzian spacetime metric and Einstein/AeST gravity from the preserved classical
    Fisher-Rao pointer sector.
  - Deriving physical rates Gamma_+/-(s) and environmental coupling eta(s) from the full 4D action.
"""

import json
import math
import pathlib
import numpy as np
from scipy.linalg import expm
import sympy as sp


def build_pointer_markov_W(M, Gp, Gm):
    W = np.zeros((M, M), dtype=float)
    for m in range(M):
        W[(m + 1) % M, m] += Gp
        W[(m - 1) % M, m] += Gm
        W[m, m] -= (Gp + Gm)
    return W


def build_full_gksl_pointer_rhs(rho, Gp, Gm, gamma_dec):
    """Exact M x M CPTP GKSL master equation RHS (no diagonal assumption)."""
    M = rho.shape[0]
    drho = np.zeros((M, M), dtype=complex)
    for m in range(M):
        mp = (m + 1) % M
        mm = (m - 1) % M
        drho[mp, mp] += Gp * rho[m, m]
        drho[m, :] -= 0.5 * Gp * rho[m, :]
        drho[:, m] -= 0.5 * Gp * rho[:, m]
        drho[mm, mm] += Gm * rho[m, m]
        drho[m, :] -= 0.5 * Gm * rho[m, :]
        drho[:, m] -= 0.5 * Gm * rho[:, m]
    drho += gamma_dec * (np.diag(np.diag(rho)) - rho)
    return drho


def gate1_literature_benchmarks():
    """Gate 1: External literature benchmarks (Hardy 2001, Breuer-Petruccione Zeno, Barato-Seifert TUR)."""
    N_F = lambda dim, b: dim + b * dim * (dim - 1) / sp.Integer(2)
    assert N_F(2, 1) - 1 == 2
    assert N_F(2, 2) - 1 == 3
    assert N_F(2, 4) - 1 == 5
    assert N_F(4, 1) - 1 == 9
    assert N_F(4, 2) - 1 == 15
    assert N_F(4, 4) - 1 == 27

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def gksl_qubit_liouvillian(Omega, gamma):
        H = 0.5 * Omega * sx
        L_ham = -1j * (np.kron(I2, H) - np.kron(H.T, I2))
        L_deph = 0.5 * gamma * (np.kron(sz.conj(), sz) - np.eye(4, dtype=complex))
        return L_ham + L_deph

    for Omega, gamma in [(1.0, 0.5), (1.0, 20.0), (1.0, 200.0)]:
        L_sup = gksl_qubit_liouvillian(Omega, gamma)
        evals_num = np.linalg.eigvals(L_sup)
        disc = (0.5 * gamma) ** 2 - Omega ** 2
        if disc >= 0:
            r = math.sqrt(disc)
            evals_lit = np.array([0.0, -gamma, -0.5 * gamma + r, -0.5 * gamma - r], dtype=complex)
        else:
            r = math.sqrt(-disc)
            evals_lit = np.array([0.0, -gamma, -0.5 * gamma + 1j * r, -0.5 * gamma - 1j * r], dtype=complex)
        idx_n = np.lexsort((evals_num.imag, evals_num.real))
        idx_l = np.lexsort((evals_lit.imag, evals_lit.real))
        assert np.max(np.abs(evals_num[idx_n] - evals_lit[idx_l])) < 1e-12

    ev_z = np.sort(np.real(np.linalg.eigvals(gksl_qubit_liouvillian(1.0, 200.0))))
    gamma_slow = float(-ev_z[-2])
    gamma_zeno_asymp = 1.0 / 200.0
    rel_zeno_err = abs(gamma_slow - gamma_zeno_asymp) / gamma_zeno_asymp
    assert rel_zeno_err < 1e-4

    for ratio in [1.001, 1.5, 3.0, 10.0]:
        tur_prod = ((ratio + 1.0) / (ratio - 1.0)) * math.log(ratio)
        assert tur_prod >= 2.0 - 1e-12
    tur_near_eq = ((1.0001 + 1.0) / (1.0001 - 1.0)) * math.log(1.0001)
    assert abs(tur_near_eq - 2.0) < 1e-6

    return {
        "hardy_local_tomography_benchmarks_exact": True,
        "gksl_zeno_rate_at_gamma200": gamma_slow,
        "gksl_zeno_asymp_rel_err": float(rel_zeno_err),
        "barato_seifert_tur_near_eq": float(tur_near_eq),
    }


def gate2_field_selection_and_pointer_clock():
    """
    Gate 2:
    (2A) Exact SymPy proof of unique selection of F = C (beta = 2) via local tomography on K_4 x K_4,
         plus explicit symbolic verification that Im(Q_{mu nu}) == 0 identically on RP^3 (odd dim 3).
    (2B) Exact M-state cyclic composite pointer-variable clock (M = 12) separating steady-state
         ticking observables (v_tick_ss, sigma_dot_ss, TUR) from transient localized-hand relaxation.
    """
    # Part 2A: Symbolic Local Tomography Proof & RP^3 Zero Symplectic Form Check
    dA, dB, beta = sp.symbols("dA dB beta", real=True)
    N_F = lambda dim, b: dim + b * dim * (dim - 1) / sp.Integer(2)
    Delta_LT = sp.factor(N_F(dA * dB, beta) - N_F(dA, beta) * N_F(dB, beta))
    expected_factor = -sp.Rational(1, 4) * dA * dB * (dA - 1) * (dB - 1) * beta * (beta - 2)
    assert sp.simplify(Delta_LT - expected_factor) == 0

    pos_sols = [s for s in sp.solve(Delta_LT.subs({dA: 4, dB: 4}), beta) if s > 0]
    assert pos_sols == [2]

    k4_defects = {
        "R_beta1": int(Delta_LT.subs({dA: 4, dB: 4, beta: 1})),
        "C_beta2": int(Delta_LT.subs({dA: 4, dB: 4, beta: 2})),
        "H_beta4": int(Delta_LT.subs({dA: 4, dB: 4, beta: 4})),
    }
    assert k4_defects == {"R_beta1": 36, "C_beta2": 0, "H_beta4": -288}

    # Explicit symbolic check: on RP^3, real state |psi(x1,x2,x3)> = (1,x1,x2,x3)^T / sqrt(1+x^2)
    # has Im(Q_{mu nu}) == 0 identically (and dim_R(RP^3) = 3 is odd => no symplectic 2-form).
    x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
    v_r = sp.Matrix([1, x1, x2, x3])
    psi_r = v_r / sp.sqrt((v_r.T * v_r)[0])
    P_r = sp.eye(4) - psi_r * psi_r.T
    d_r = [psi_r.diff(v) for v in (x1, x2, x3)]
    for i in range(3):
        for j in range(3):
            q_ij = sp.simplify((d_r[i].T * P_r * d_r[j])[0])
            assert sp.im(q_ij) == 0, f"RP^3 QGT not purely real at ({i},{j}): {q_ij}"

    # Part 2B: Exact M-State Composite Pointer-Variable Clock (M = 12)
    M = 12
    dists = np.array([min(m, M - m) for m in range(M)], dtype=float)
    p0 = np.exp(-0.5 * (dists / 1.2) ** 2)
    p0 /= np.sum(p0)
    rho0_diag = np.diag(p0.astype(complex))

    Gp_ref, Gm_ref = 2.5, 0.5
    rhs_0 = build_full_gksl_pointer_rhs(rho0_diag, Gp_ref, Gm_ref, 0.0)
    rhs_1e6 = build_full_gksl_pointer_rhs(rho0_diag, Gp_ref, Gm_ref, 1e6)
    qnd_immunity_err = float(np.max(np.abs(rhs_0 - rhs_1e6)))
    assert qnd_immunity_err == 0.0

    W_ref = build_pointer_markov_W(M, Gp_ref, Gm_ref)
    assert np.max(np.abs(np.real(np.diag(rhs_0)) - (W_ref @ p0))) < 1e-15

    # Steady-state uniform marginal p_ss = 1/M has dp_ss/dt = 0 for all A, while v_tick_ss = G+ - G-
    # and sigma_dot_ss = (G+ - G-) A vanish iff A = 0.
    p_ss = np.full(M, 1.0 / M)
    G_tot = 2.0
    affinity_scan = []
    for A in [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]:
        Gp = G_tot * math.exp(A) / (1.0 + math.exp(A))
        Gm = G_tot / (1.0 + math.exp(A))
        W = build_pointer_markov_W(M, Gp, Gm)
        assert float(np.max(np.abs(W @ p_ss))) < 1e-15

        # Transient localized pointer-hand evolution at t = 0.25 (A=0 is pure diffusive relaxation)
        p_t = expm(W * 0.25) @ p0
        dp_t = W @ p_t
        v_FR_trans = math.sqrt(0.25 * float(np.sum((dp_t ** 2) / p_t)))
        sigma_dot_trans = 0.0
        for m in range(M):
            mp = (m + 1) % M
            j_f = Gp * p_t[m]
            j_b = Gm * p_t[mp]
            if j_f > 0 and j_b > 0:
                sigma_dot_trans += (j_f - j_b) * math.log(j_f / j_b)

        v_tick_ss = Gp - Gm
        sigma_ss = (Gp - Gm) * A
        tur_ss = ((Gp + Gm) / (Gp - Gm)) * A if A > 0 else 2.0
        assert tur_ss >= 2.0 - 1e-12
        affinity_scan.append({
            "affinity_A": A,
            "Gamma_plus": round(Gp, 6),
            "Gamma_minus": round(Gm, 6),
            "steady_v_tick": round(v_tick_ss, 6),
            "steady_sigma_dot": round(sigma_ss, 6),
            "steady_tur_product": round(tur_ss, 6),
            "transient_fisher_rao_speed_t025": round(v_FR_trans, 6),
            "transient_sigma_dot_t025": round(sigma_dot_trans, 6),
        })

    return {
        "local_tomography_defect_formula": str(Delta_LT),
        "unique_positive_beta_solution": 2,
        "rp3_qgt_imaginary_part_identically_zero": True,
        "rp3_real_dim_odd_no_symplectic_form": 3,
        "k4_x_k4_local_tomography_defects": k4_defects,
        "pointer_clock_M_ticks": M,
        "pointer_clock_qnd_immunity_err_gamma_1e6": qnd_immunity_err,
        "steady_state_ring_marginal_fisher_rao_speed_all_A": 0.0,
        "affinity_scan": affinity_scan,
    }


def gate3_invariance_and_limit_tests():
    """Gate 3: Full GKSL trace/Hermiticity/off-diagonal decay, positivity, and arrow reversal symmetry."""
    M = 12
    Gp, Gm, gamma_dec = 2.2, 0.6, 50.0
    dists = np.array([min(m, M - m) for m in range(M)], dtype=float)
    p0 = np.exp(-0.5 * (dists / 1.2) ** 2)
    p0 /= np.sum(p0)

    # 1. Probability conservation & positivity for t = 0.5, 2.0, 10.0
    W_fwd = build_pointer_markov_W(M, Gp, Gm)
    for t_val in [0.5, 2.0, 10.0]:
        pt = expm(W_fwd * t_val) @ p0
        assert abs(float(np.sum(pt)) - 1.0) < 1e-14
        assert float(np.min(pt)) >= -1e-15

    # 2. Time-arrow reversal symmetry (Gp <-> Gm)
    W_rev = build_pointer_markov_W(M, Gm, Gp)
    p_fwd = expm(W_fwd * 0.25) @ p0
    p_rev = expm(W_rev * 0.25) @ p0
    p_fwd_mirror = np.array([p_fwd[(-m) % M] for m in range(M)])
    rev_sym_err = float(np.max(np.abs(p_rev - p_fwd_mirror)))
    assert rev_sym_err < 1e-14

    v_FR_fwd = math.sqrt(0.25 * float(np.sum(((W_fwd @ p_fwd) ** 2) / p_fwd)))
    v_FR_rev = math.sqrt(0.25 * float(np.sum(((W_rev @ p_rev) ** 2) / p_rev)))
    assert abs(v_FR_fwd - v_FR_rev) < 1e-14

    # 3. Full M x M GKSL trace preservation, Hermiticity, and exact off-diagonal decay rate
    rho_coh = np.diag(p0.astype(complex))
    rho_coh[0, 1] = 0.05 + 0.02j
    rho_coh[1, 0] = 0.05 - 0.02j
    drho = build_full_gksl_pointer_rhs(rho_coh, Gp, Gm, gamma_dec)
    trace_err = float(abs(np.trace(drho)))
    herm_err = float(np.max(np.abs(drho - drho.conj().T)))
    assert trace_err < 1e-14, f"GKSL trace not preserved: {trace_err}"
    assert herm_err < 1e-14, f"GKSL Hermiticity violated: {herm_err}"
    # Exact off-diagonal decay rate for rho_{01}: -(Gp + Gm + gamma_dec) * rho_{01}
    expected_offdiag_rate = -(Gp + Gm + gamma_dec) * rho_coh[0, 1]
    offdiag_rate_err = float(abs(drho[0, 1] - expected_offdiag_rate))
    assert offdiag_rate_err < 1e-13, f"Off-diagonal rate mismatch: {offdiag_rate_err}"
    # Diagonal evolution is strictly unaffected by the off-diagonal coherence or gamma_dec
    diag_indep_err = float(np.max(np.abs(np.real(np.diag(drho)) - (W_fwd @ p0))))
    assert diag_indep_err < 1e-14

    return {
        "probability_conservation_and_positivity": True,
        "arrow_reversal_mirror_err": rev_sym_err,
        "fisher_rao_reversal_symmetry_err": float(abs(v_FR_fwd - v_FR_rev)),
        "full_gksl_trace_err": trace_err,
        "full_gksl_hermiticity_err": herm_err,
        "full_gksl_offdiag_decay_rate_err": offdiag_rate_err,
        "full_gksl_diag_independence_err": diag_indep_err,
    }


def main():
    g1 = gate1_literature_benchmarks()
    g2 = gate2_field_selection_and_pointer_clock()
    g3 = gate3_invariance_and_limit_tests()

    out = {
        "script": "verify_qdec_field_selection_and_pointer_clock.py",
        "epistemic_status": {
            "NATURE_LEVEL_CLOSURE": False,
            "PROVEN_ALGEBRAIC": [
                "Among the three Frobenius associative division algebras F in {R, C, H} (beta in {1, 2, 4}), the compositional local tomography defect is identically Delta_LT(d_A, d_B, beta) = -(1/4) d_A d_B (d_A - 1)(d_B - 1) beta (beta - 2). For any composite system d_A, d_B >= 2 (K_4 x K_4: Delta_R = +36, Delta_C = 0, Delta_H = -288 under naive Herm(d_A d_B, H) since non-commutative H lacks a standard tensor product), local tomography holds IF AND ONLY IF F = C (beta = 2). Furthermore, on RP^3 the canonical QGT imaginary part vanishes identically (Im Q_{mu nu} = 0) and dim_R(RP^3) = 3 is odd, admitting no symplectic 2-form.",
                "A coherent single-qubit clock H = (Omega/2) sigma_x under strong pointer-basis dephasing gamma >> 2*Omega suffers exact Quantum Zeno suppression Gamma_Zeno = Omega^2 / gamma -> 0 (verified against the 4x4 GKSL Liouvillian spectrum).",
                "A composite M-state cyclic pointer-variable clock (diagonal in the decoherence-free pointer basis {|m>}) is 100% invariant under arbitrary pointer-basis QND dephasing gamma_dec in [0, infinity) (error = 0.0 at gamma_dec = 10^6; full M x M GKSL trace and Hermiticity verified to < 1e-14). In steady state (uniform ring marginal p^{ss} = 1/M, where periodic marginal ds_FR/dt = 0 for all A), net tick velocity v_tick,ss = Gamma_+ - Gamma_- and entropy production sigma_dot,ss = (Gamma_+ - Gamma_-) A are strictly positive iff A = ln(Gamma_+/Gamma_-) != 0 (obeying Barato-Seifert TUR >= 2) and vanish iff A = 0; for a localized transient pointer hand, A = 0 yields pure diffusive relaxation while A > 0 adds directed drift to transient Fisher-Rao speed and entropy production.",
            ],
            "OPEN_QDEC_TDEC": [
                "Why Nature enforces compositional local tomography at the pregeometric K_4 seed (turning F = C from a conditional axiom into a dynamical theorem).",
                "Deriving 4D Lorentzian spacetime metric and Einstein/AeST gravity from the preserved classical Fisher-Rao pointer sector.",
                "Deriving the microscopic transition rates Gamma_+/-(s) and environmental coupling eta(s) from the full 4D non-perturbative Tau Core action.",
            ],
        },
        "gate1_benchmarks": g1,
        "gate2_derivation": g2,
        "gate3_invariances": g3,
    }

    out_path = pathlib.Path(__file__).with_name("qdec_field_selection_and_pointer_clock_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
