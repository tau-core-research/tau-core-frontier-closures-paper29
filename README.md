# Tau Core Frontier Closures — Paper 29 (`v1.1.0-step1`)

**Author:** Jozsef Olcsak (ORCID: [0009-0003-7311-6522](https://orcid.org/0009-0003-7311-6522))  
**Date:** 23 November 2026  
**Repository Visibility Policy:** Strictly `PRIVATE` until Zenodo DOI minting.

## Scope and Three-Pillar Architecture

Paper 29 addresses the three conditional/open theoretical frontiers identified at the completion of Paper 28 (`v4.5.0`):

1. **Pillar 1 (`Step 1` — Implemented & Verified in `v1.1.0-step1`):** Covariant 4D disformal clock-lapse optical resolution of the $+0.271\%$ $\theta_*$–`DESI` BAO–SN triangle (`Gates S1.1–S1.4`), exact 4D relativistic Swiss-Cheese LTB Weinberg No-Go theorem, exact `SymPy` + `DOP853` 4D Sachs optical identity and Null Convergence Condition, zero-parameter $\alpha \equiv \alpha_{\mathrm{OCSR}} = 5\sqrt{3}/24$ confrontation across untruncated $\ell \in [2, 3000]$ `ACT DR6` + `ACT+PR4` CMB lensing + `DESI` BAO + `CC15` + 4 SN Ia compilations, and the $1341\times 1341$ `Pantheon+SH0ES` local-void Hubble ceiling.
2. **Pillar 2 (`Step 2` — Scheduled Next):** Nonlinear 3D colliding galaxy cluster (`1E 0657-558`, Bullet Cluster) hydrodynamics + collisionless `AeST` clock-condensate + nonlinear QUMOND–Helmholtz weak-lensing $\kappa(\vec{\theta})$ audit ($\sum m_\nu = 0.05919\;\mathrm{eV}$, $k_{\mathrm{added}} = 0$).
3. **Pillar 3 (`Step 3` — Scheduled After Pillar 2):** Microscopic 4D $D_4$ (24-cell) simplicial lattice partition function and continuum EFT derivation of the covariant `AeST`–QUMOND–Helmholtz action and macroscopic OCSR weight $\alpha_{\mathrm{OCSR}}$.

## Reproducing Pillar 1 (`Step 1`) Under Referee Discipline (`python3 -W error`)

```bash
python3 -W error scripts/verify_paper29_step1_theta_h0_void_optical_closure.py
python3 -W error scripts/generate_paper29_step1_figures.py
pdflatex -interaction=nonstopmode Tau_Core_Paper_29_Frontier_Closures_EN.tex
pdflatex -interaction=nonstopmode Tau_Core_Paper_29_Frontier_Closures_HU.tex
```

## Pillar 1 Epistemic Summary (`paper29_step1_verification_results.json`)

| Gate | Test | Epistemic Tag | Verified Result |
| :--- | :--- | :--- | :--- |
| `S1.1` | 4D $C^2$ Darmois–Israel LTB Swiss-Cheese ergodic sightlines & Ricci/Weyl Sachs | `PROVEN NO-GO` | $\langle\Delta D_A(z_*)/D_A^{\mathrm{FLRW}}(z_*)\rangle_\Omega = +0.00265\%$ ($102.4\times$ shortfall vs. $0.271\%$) |
| `S1.2` | 4D disformal clock-lapse metric $\tilde{g}_{\mu\nu}^{\mathrm{opt}}$ `SymPy` + `DOP853` Sachs & NCC | `PROVEN` | Symbolic residual $\equiv 0$; `DOP853` error $2.65\times 10^{-13}$; $\min_z \mathrm{d}E_{\mathrm{opt}}/\mathrm{d}z = 0.58001 > 0$ |
| `S1.3a` | Zero-param. $\alpha = \alpha_{\mathrm{OCSR}} = 5\sqrt{3}/24$ $\ell \le 3000$ CMB + BAO + CC15 + 4 SNe | `PROVEN` | $100\theta_* = 1.04090$ at $H_0 = 67.0708$; `ACT108` $1.008/\mathrm{dof}$; `DESI13` $\chi^2: 33.17 \to 11.43$ |
| `S1.3b` | Microscopic 4D lattice derivation of unit Wiltshire weight on $\alpha_{\mathrm{OCSR}}$ | `CONDITIONAL` | $-1.21\sigma$ from Wilks $\hat{\alpha} = 0.4769 \pm 0.0958$; $\Delta\mathrm{BIC} = -5.91$ vs. floated $\alpha$ |
| `S1.4` | Official $1341\times 1341$ `Pantheon+SH0ES` local-void Hubble ceiling | `PROVEN` Ceiling | $H_0^{\mathrm{loc}} \le 68.326\;\mathrm{km\,s^{-1}\,Mpc^{-1}}$ ($-0.65\sigma$ JWST TRGB; `SH0ES` needs $\Delta M_B = 0.145\;\mathrm{mag}$) |
