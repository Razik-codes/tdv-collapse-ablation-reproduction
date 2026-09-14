# Contributions and scope

## Independent work by Razik-codes

This repository is an independent, downscaled reproduction artifact. The repository owner contributed:

- the low-rank-motion synthetic proxy in `data/cv/generate_synthetic_motion.py`;
- the three-arm, single-GPU reproduction harness in `job_scripts/pretrain_tdv_local_smoketest.sh`;
- the interactive marimo tutorial in `claim_tutorial.py`;
- the collapse-ablation report, figures, result interpretation, and limitations in `reports/collapse-ablation/`;
- the publication-facing documentation, citation metadata, and source-provenance records.

## What is upstream or third party

The TDV model, its original training framework, data loaders, evaluation integration, and bundled projects are upstream or adapted third-party work. They are included to make the reproduction inspectable; this artifact does not claim authorship of them. See `THIRD_PARTY_NOTICES.md` for component-level provenance and licensing.

## Experimental record

The public launcher on `main` consolidates the recorded control and ablation configurations and exposes the same three arms through `TDV_ARM=full`, `TDV_ARM=no-motion`, and `TDV_ARM=no-mse`. Results reported in the tutorial and report are qualitative, mechanism-level evidence from a downscaled synthetic proxy, not a reproduction of the original SSv2 and ImageNet benchmark numbers.

## Attribution

When discussing TDV itself, cite the original paper. When discussing this downscaled artifact, cite this repository using `CITATION.cff`.
