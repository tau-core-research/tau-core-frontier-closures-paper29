# Tau Core Frontier Closures — Paper 29 (`v1.2.0-step2`)

**Author:** Jozsef Olcsak (ORCID: [0009-0003-7311-6522](https://orcid.org/0009-0003-7311-6522))  
**Date:** 23 November 2026  
**Repository Visibility Policy:** Strictly `PRIVATE` until Zenodo DOI minting.

## Scope and Three-Pillar Architecture

Paper 29 addresses the three conditional/open theoretical frontiers identified at the completion of Paper 28 (`v4.5.0`):

1. **Pillar 1 (`Step 1` — Certified & Verified in `v1.1.0-step1`):** Covariant 4D disformal clock-lapse optical resolution of the $+0.271\%$ $\theta_*$–`DESI` BAO–SN triangle (`Gates S1.1–S1.4`), exact 4D relativistic Swiss-Cheese LTB Weinberg No-Go theorem, exact `SymPy` + `DOP853` 4D Sachs optical identity and Null Convergence Condition, zero-parameter $\alpha \equiv \alpha_{\mathrm{OCSR}} = 5\sqrt{3}/24$ confrontation across untruncated $\ell \in [2, 3000]$ `ACT DR6` + `ACT+PR4` CMB lensing + `DESI` BAO + `CC15` + 4 SN Ia compilations, and the $1341\times 1341$ `Pantheon+SH0ES` local-void Hubble ceiling.
2. **Pillar 2 (`Step 2` — Implemented & Verified in `v1.2.0-step2`):** 3+1D director-expulsion, isolated $1\text{--}1.5\;\mathrm{Mpc}$ $L_*$ galaxy $\psi_{\mathrm{IR}}$-tail preservation & variational two-sector Noether momentum theorems, Pauli-waterbag & Maxwellian Tremaine–Gunn $\sum m_\nu = 0.05919\;\mathrm{eV}$ active-neutrino & baryon-only 3D QUMOND No-Go theorems, 3D Cartesian symplectic violent-relaxation phase-space core formation at identical $t=8.0$ ($\sigma_0/V_{\mathrm{col}} = \sqrt{c_{\mathrm{mix}}/3} = 0.24115$), three genuinely live fixed-$R_{200}$ 3D `MUSCL–HLL` hydrodynamical + $40{,}000$-particle N-body + isolated Hockney–Eastwood variational QUMOND merger simulations of `1E 0657-558` (Bullet Cluster), and global-$R_{200}$, $h$-harmonized 64-aperture out-of-sample cross-scale validation across `12 X-COP` and `20 CLASH` clusters (`Gates S2.1–S2.4`).
3. **Pillar 3 (`Step 3` — Scheduled Next):** Microscopic 4D $D_4$ (24-cell) simplicial lattice partition function and continuum EFT derivation of the covariant `AeST`–QUMOND–Helmholtz action and macroscopic OCSR weight $\alpha_{\mathrm{OCSR}}$.

## Reproducing Pillars 1 & 2 Under Referee Discipline (`python3 -W error`)

```bash
python3 -W error scripts/verify_paper29_step1_theta_h0_void_optical_closure.py
python3 -W error scripts/generate_paper29_step1_figures.py
python3 -W error scripts/verify_paper29_step2_bullet_cluster_3d_hydro_qumond.py
python3 -W error scripts/generate_paper29_step2_figures.py
pdflatex -interaction=nonstopmode Tau_Core_Paper_29_Frontier_Closures_EN.tex
pdflatex -interaction=nonstopmode Tau_Core_Paper_29_Frontier_Closures_HU.tex
```

## Epistemic Summary (`paper29_step1_verification_results.json` & `paper29_step2_verification_results.json`)

| Gate | Test | Epistemic Tag | Verified Result |
| :--- | :--- | :--- | :--- |
| `S1.1` | 4D $C^2$ Darmois–Israel LTB Swiss-Cheese ergodic sightlines & Ricci/Weyl Sachs | `PROVEN NO-GO` | $\langle\Delta D_A(z_*)/D_A^{\mathrm{FLRW}}(z_*)\rangle_\Omega = +0.00265\%$ ($102.4\times$ shortfall vs. $0.271\%$) |
| `S1.2` | 4D disformal clock-lapse metric $\tilde{g}_{\mu\nu}^{\mathrm{opt}}$ `SymPy` + `DOP853` Sachs & NCC | `PROVEN` | Symbolic residual $\equiv 0$; `DOP853` error $2.65\times 10^{-13}$; $\min_z \mathrm{d}E_{\mathrm{opt}}/\mathrm{d}z = 0.58001 > 0$ |
| `S1.3a` | Zero-param. $\alpha = \alpha_{\mathrm{OCSR}} = 5\sqrt{3}/24$ $\ell \le 3000$ CMB + BAO + CC15 + 4 SNe | `PROVEN` | $100\theta_* = 1.04090$ at $H_0 = 67.0708$; `ACT108` $1.008/\mathrm{dof}$; `DESI13` $\chi^2: 33.17 \to 11.43$ |
| `S1.3b` | Microscopic 4D lattice derivation of unit Wiltshire weight on $\alpha_{\mathrm{OCSR}}$ | `CONDITIONAL` | $-1.21\sigma$ from Wilks $\hat{\alpha} = 0.4769 \pm 0.0958$; $\Delta\mathrm{BIC} = -5.91$ vs. floated $\alpha$ |
| `S1.4` | Official $1341\times 1341$ `Pantheon+SH0ES` local-void Hubble ceiling | `PROVEN` Ceiling | $H_0^{\mathrm{loc}} \le 68.326\;\mathrm{km\,s^{-1}\,Mpc^{-1}}$ ($-0.65\sigma$ JWST TRGB; `SH0ES` needs $\Delta M_B = 0.145\;\mathrm{mag}$) |
| `S2.1a` | 3+1D director expulsion, $1\;\mathrm{Mpc}$ $\psi_{\mathrm{IR}}$-tail & variational Noether (`SymPy`) | `PROVEN` | $m_{H,\mathrm{loc}}/m_{H,\infty} = 3.01\times 10^{-8}$; $1\;\mathrm{Mpc}$ $\psi_{\mathrm{IR}}$: $55.57 \to 69.67\;\mathrm{km\,s^{-1}}$ |
| `S2.1b` | Pauli-waterbag & Maxwellian Tremaine–Gunn $\sum m_\nu = 0.0592\;\mathrm{eV}$ bounds | `PROVEN NO-GO` | Pauli $1.05\times 10^8\,M_\odot$ ($2.67\times 10^6\times$ shortfall); Maxw. $3.53\times 10^7\,M_\odot$ ($7.93\times 10^6\times$) |
| `S2.2` | 3D Cartesian symplectic violent relaxation at $t=8.0$ ($\sigma_0/V_{\mathrm{col}} = \sqrt{c_{\mathrm{mix}}/3}$) | `PROVEN` | Pre-heated core slope $-0.5318$ vs. cold cusp $-1.9095$ (bound Clausius virial $0.980, 0.978$) |
| `S2.3a` | Live fixed-$R_{200}$ 3D `MUSCL–HLL` + $40\mathrm{k}$ N-body + variational QUMOND | `PROVEN` / `NO-GO` | $\mathcal{M}=2.67$, CoM $8.29\times 10^{-5}$; `M3a` $(143.0, 251.1)\;\mathrm{kpc}$, BCG lag $13.7\;\mathrm{kpc}$; `M0a` $-5.13\sigma$ / $-6.70\sigma$ |
| `S2.3b` | Live fixed-$R_{200}$ $250\;\mathrm{kpc}$ mass under $+1\sigma / +1.5\sigma$ $c_{200}$ scatter | `CONDITIONAL` | Live `M3a`: $(2.102, 1.312)$; live $+1\sigma$ `M3b`: $(2.409, 1.316)$; live `M3c`: $(2.593, 1.328)\times 10^{14}\,M_\odot$ |
| `S2.4` | Global-$R_{200}$, $h$-harmonized 64-aperture `X-COP` + `CLASH` ($r_{2500}^X$, BCG) audit | `PROVEN` | `XCOP24` $\chi^2/\mathrm{dof}=0.872$ ($w_{\mathrm{GC}}$) / $1.398$ (EP); $r_{2500}^X$ $0.708$; BCG $\chi^2/20: 15.57 \to 2.71$ |
