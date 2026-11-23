# Tau Core Frontier Closures — Paper 29 (`v2.0.0` — Complete Three-Pillar Edition)

**Author:** Jozsef Olcsak (ORCID: [0009-0003-7311-6522](https://orcid.org/0009-0003-7311-6522))  
**Date:** 23 November 2026  
**Repository Visibility Policy:** Strictly `PRIVATE` until Zenodo DOI minting.

## Scope and Three-Pillar Architecture

Paper 29 closes the three conditional/open theoretical frontiers identified at the completion of Paper 28 (`v4.5.0`) under strict *PRD / JCAP / ApJ / CMP / CQG* referee discipline (`python3 -W error`, zero silent clamps on physical observables, zero added dark-sector parameters $k_{\mathrm{added}} = 0$):

1. **Pillar 1 (`Step 1` — Certified `ACCEPT AS IS` in `v1.1.0-step1`):** Covariant 4D disformal clock-lapse optical resolution of the $+0.271\%$ $\theta_*$–`DESI` BAO–SN triangle (`Gates S1.1–S1.4`), exact 4D relativistic Swiss-Cheese LTB Weinberg No-Go theorem, exact `SymPy` + `DOP853` 4D Sachs optical identity and Null Convergence Condition, zero-parameter $\alpha \equiv \alpha_{\mathrm{OCSR}} = 5\sqrt{3}/24$ confrontation across untruncated $\ell \in [2, 3000]$ `ACT DR6` + `ACT+PR4` CMB lensing + `DESI` BAO + `CC15` + 4 SN Ia compilations, and the $1341\times 1341$ `Pantheon+SH0ES` local-void Hubble ceiling.
2. **Pillar 2 (`Step 2` — Certified `ACCEPT AS IS` in `v1.2.0-step2`):** 3+1D director-expulsion, isolated $1\text{--}1.5\;\mathrm{Mpc}$ $L_*$ galaxy $\psi_{\mathrm{IR}}$-tail preservation & variational two-sector Noether momentum theorems, Pauli-waterbag & Maxwellian Tremaine–Gunn $\sum m_\nu = 0.05919\;\mathrm{eV}$ active-neutrino & baryon-only 3D QUMOND No-Go theorems, 3D Cartesian symplectic violent-relaxation phase-space core formation at identical $t=8.0$ ($\sigma_0/V_{\mathrm{col}} = \sqrt{c_{\mathrm{mix}}/3} = 0.24115$), three genuinely live fixed-$R_{200}$ 3D `MUSCL–HLL` hydrodynamical + $40{,}000$-particle N-body + isolated Hockney–Eastwood variational QUMOND merger simulations of `1E 0657-558` (Bullet Cluster), and global-$R_{200}$, $h$-harmonized 64-aperture out-of-sample cross-scale validation across `12 X-COP` and `20 CLASH` clusters (`Gates S2.1–S2.4`).
3. **Pillar 3 (`Step 3` — Complete Microscopic-to-Continuum Closure in `v2.0.0`):** Exact 4D 24-cell ($W(F_4)$, $1{,}152$ elements) tensor-ring equivalence to continuous $O(4)$ through rank $r = 5$ (eliminating all $d \le 7$ LIV to all loops), explicit $\mathrm{Pin}(W(F_4))$ Overlap Dirac covariance, 32-triangle single-slice Osterwalder–Schrader positivity & $d_S(\sigma) \to 4$, local $S_4(\sigma)\times S_4(\sigma')$ Reynolds vanishing $\kappa_{\mathrm{dir}}^{\mathrm{bare}} \equiv 0$ curing the $7.40\times 10^{13}\times$ Planck-stiffness freezing hierarchy, zero-parameter symbolic derivation of $(\alpha_{\mathrm{OCSR}}, \kappa_\theta, c_2, R_{A/b})$ from $u = \sqrt{3}/24$, 60-decade Wetterich–Frullani functional RG & $196{,}608$-tetrahedron 3D Kuhn-simplicial $\mathcal{O}(a^2)$ $\Gamma$-convergence, $64^3$ cosmic-web homogenization of $N_{\mathrm{clock}}(z)$, and the complete irreducible UV epistemic ledger (`Gates S3.1–S3.4`).

## End-to-End Reproduction Under Referee Discipline (`python3 -W error`)

```bash
python3 -W error scripts/verify_paper29_step1_theta_h0_void_optical_closure.py
python3 -W error scripts/generate_paper29_step1_figures.py
python3 -W error scripts/verify_paper29_step2_bullet_cluster_3d_hydro_qumond.py
python3 -W error scripts/generate_paper29_step2_figures.py
python3 -W error scripts/verify_paper29_step3_24cell_lattice_continuum_eft.py
python3 -W error scripts/generate_paper29_step3_figures.py
pdflatex -interaction=nonstopmode Tau_Core_Paper_29_Frontier_Closures_EN.tex
pdflatex -interaction=nonstopmode Tau_Core_Paper_29_Frontier_Closures_HU.tex
```

## Complete Three-Pillar Epistemic Ledger (`paper29_step{1,2,3}_verification_results.json`)

| Gate / Axiom | Test or Irreducible Boundary | Epistemic Tag | Verified Quantitative Result |
| :--- | :--- | :--- | :--- |
| `S1.1` | 4D $C^2$ Darmois–Israel LTB Swiss-Cheese ergodic sightlines & Ricci/Weyl Sachs | `PROVEN NO-GO` | $\langle\Delta D_A(z_*)/D_A^{\mathrm{FLRW}}(z_*)\rangle_\Omega = +0.00265\%$ ($102.4\times$ shortfall vs. $0.271\%$) |
| `S1.2` | 4D disformal clock-lapse metric $\tilde{g}_{\mu\nu}^{\mathrm{opt}}$ `SymPy` + `DOP853` Sachs & NCC | `PROVEN` | Symbolic residual $\equiv 0$; `DOP853` error $2.65\times 10^{-13}$; $\min_z \mathrm{d}E_{\mathrm{opt}}/\mathrm{d}z = 0.58001 > 0$ |
| `S1.3a` | Zero-param. $\alpha = \alpha_{\mathrm{OCSR}} = 5\sqrt{3}/24$ $\ell \le 3000$ CMB + BAO + CC15 + 4 SNe | `PROVEN` | $100\theta_* = 1.04090$ at $H_0 = 67.0708$; `ACT108` $1.008/\mathrm{dof}$; `DESI13` $\chi^2: 33.17 \to 11.43$ |
| `S1.3b` | Unit Wiltshire weight on $\alpha_{\mathrm{OCSR}}$ (closed in `Gate S3.4`) | `PROVEN` / `COND.` | Two-domain unit weight $\equiv 1$; $-1.21\sigma$ from Wilks $\hat{\alpha} = 0.4769 \pm 0.0958$; $\Delta\mathrm{BIC} = -5.91$ |
| `S1.4` | Official $1341\times 1341$ `Pantheon+SH0ES` local-void Hubble ceiling | `PROVEN` Ceiling | $H_0^{\mathrm{loc}} \le 68.326\;\mathrm{km\,s^{-1}\,Mpc^{-1}}$ ($-0.65\sigma$ JWST TRGB; `SH0ES` needs $\Delta M_B = 0.145\;\mathrm{mag}$) |
| `S2.1a` | 3+1D director expulsion, $1\;\mathrm{Mpc}$ $\psi_{\mathrm{IR}}$-tail & variational Noether (`SymPy`) | `PROVEN` | $m_{H,\mathrm{loc}}/m_{H,\infty} = 3.01\times 10^{-8}$; $1\;\mathrm{Mpc}$ $\psi_{\mathrm{IR}}$: $55.57 \to 69.67\;\mathrm{km\,s^{-1}}$ |
| `S2.1b` | Pauli-waterbag & Maxwellian Tremaine–Gunn $\sum m_\nu = 0.0592\;\mathrm{eV}$ bounds | `PROVEN NO-GO` | Pauli $1.05\times 10^8\,M_\odot$ ($2.67\times 10^6\times$ shortfall); Maxw. $3.53\times 10^7\,M_\odot$ ($7.93\times 10^6\times$) |
| `S2.2` | 3D Cartesian symplectic violent relaxation at $t=8.0$ ($\sigma_0/V_{\mathrm{col}} = \sqrt{c_{\mathrm{mix}}/3}$) | `PROVEN` | Pre-heated core slope $-0.5318$ vs. cold cusp $-1.9095$ (bound Clausius virial $0.980, 0.978$) |
| `S2.3a` | Live fixed-$R_{200}$ 3D `MUSCL–HLL` + $40\mathrm{k}$ N-body + variational QUMOND | `PROVEN` / `NO-GO` | $\mathcal{M}=2.67$, CoM $8.29\times 10^{-5}$; `M3a` $(143.0, 251.1)\;\mathrm{kpc}$, BCG lag $13.7\;\mathrm{kpc}$; `M0a` $-5.13\sigma$ / $-6.70\sigma$ |
| `S2.3b` | Live fixed-$R_{200}$ $250\;\mathrm{kpc}$ mass under $+1\sigma / +1.5\sigma$ $c_{200}$ scatter | `CONDITIONAL` | Live `M3a`: $(2.102, 1.312)$; live $+1\sigma$ `M3b`: $(2.409, 1.316)$; live `M3c`: $(2.593, 1.328)\times 10^{14}\,M_\odot$ |
| `S2.4` | Global-$R_{200}$, $h$-harmonized 64-aperture `X-COP` + `CLASH` ($r_{2500}^X$, BCG) audit | `PROVEN` | `XCOP24` $\chi^2/\mathrm{dof}=0.872$ ($w_{\mathrm{GC}}$) / $1.398$ (EP); $r_{2500}^X$ $0.708$; BCG $\chi^2/20: 15.57 \to 2.71$ |
| `S3.1a` | Rigid 3D $+K_4$ ($d=5$) & hypercubic $W(B_4)$ ($d=6$) LIV falsification | `PROVEN NO-GO` | Rigid $+K_4$ excluded $86.83\times$ by LHAASO; $W(B_4)$ rank-4 invariants $4 \ne 3$ |
| `S3.1b` | 4D 24-cell $W(F_4)$ $r\le 5$ isotropy, $\mathrm{Pin}(4)$ Overlap, OS positivity & $d_S \to 4$ | `PROVEN` | $d\le 7$ LIV $\equiv 0$ (all loops); LHAASO $d=8$ $|\Delta v_g/c| = 6.72\times 10^{-56}$; $d_S(100) = 4.00502$ |
| `S3.2` | Discrete $K_4$ defect algebra, local $S_4\times S_4$ Planck shield & coupling chain | `PROVEN` | $\mathcal{P}_{S_4\times S_4} \equiv 0$ (cures $7.40\times 10^{13}\times$ trap); $c_2 = 1.722\times 10^{-10}$, $c_S^2 \equiv 1$, $\alpha_2^{\mathrm{PPN}} \equiv 0$ |
| `S3.3` | 60-decade Wetterich–Frullani RG & $196{,}608$-tet 3D Kuhn-simplicial $\Gamma$-limit | `PROVEN` | 60-dec err $8.88\times 10^{-16}$; $\lambda_\parallel \ge 0.94119$; $\mathcal{O}(a^2)$ ratios $(4.66, 4.27)$; $F_{\mathrm{net}} = 4.35\times 10^{-17}$ |
| `S3.4` | $64^3$ cosmic-web $N_{\mathrm{clock}}(z)$ homogenization & boundary-layer audit | `PROVEN` / `COND.` | Two-domain unit weight $\equiv 1$; 512-ray err $0.23\%$; smooth/tracker $+0.02\% / +7.63\%$ vs. $\pm 26.5\%$ |
| `AXIOM-OCC/KMS` | Occupied $|\nu|=1$ $K_4 \to 24$-cell Lorentzian measure & horizon KMS floor | `FOUND. LAW` | Irreducible UV quantum-gravity axioms ($r_{\mathrm{seed}} = 1/4$, $a_0^{\mathrm{KH}}(z)$) + `AXIOM-FRG` ($s\sim 1$) |
