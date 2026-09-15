# DR-SMAA reproducibility package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22774719.svg)](https://doi.org/10.5281/zenodo.22774719)

Reproducibility materials for **Distributionally Robust Stochastic Multicriteria Acceptability Analysis under Probability-Model Ambiguity: An Application to Frontier AI Governance**.

## What is authoritative

This repository intentionally excludes historical/intermediate outputs that preceded the mathematical audit. The authoritative pipeline is:

1. `src/01_synthetic_benchmark.py` — calibrated synthetic benchmark with **correct fixed-support TV event bounds** and a clearly labeled **cluster-aggregated Wasserstein sensitivity approximation**.
2. `src/02_contamination_analysis.py` — recomputes contamination metrics and Wilson intervals from the frozen 5,400-replication synthetic dataset.
3. `src/03_ai_governance_publicdata.py` — reconstructs the public-data 4×6 matrix from extracted public statistics and policy-feature coding, then reruns SMAA/DR-SMAA.
4. `src/run_all.py` — runs all three analyses.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r environment/requirements.txt
python src/run_all.py
```

Validated with Python 3.12.14; the original package was also tested with Python 3.13.5. The scripts are deterministic where seeds are specified.

## Repository layout

- `data/synthetic/` — all parameters needed to regenerate the synthetic benchmark.
- `data/contamination/` — frozen directed path and 5,400 random-contamination replications used as primary synthetic input for validation.
- `data/governance/` — extracted public statistics, policy-feature coding, and 24-cell audit trail.
- `results/` — archived final paper results plus regenerated results.
- `figures/` — paper figures and regenerated checks.
- `src/` — executable analysis code.
- `docs/` — data dictionary, source provenance, table/figure mapping, and reproducibility notes.

## Important methodological notes

- Fixed-support TV reweighting cannot create rank/safety events absent from the retained support. Thus empty events have lower=upper=0 and full-support events lower=upper=1 for every TV radius.
- Wasserstein numbers in the synthetic study are **cluster-aggregated sensitivity results**. They are exact for the aggregated surrogate but **not** exact lower/upper probabilities of the original binary rank event over all 30,000 scenarios.
- The historical raw contamination files contain a column named `false_robustness`; in the final paper this quantity is correctly interpreted and reported as **oracle disagreement**. The raw file is preserved unchanged for provenance.
- In the AI-governance application, `g2` is intentionally non-discriminating. A single `g2` realization is shared across all alternatives within each Monte Carlo replication.

## Public-source policy

Third-party webpages/PDFs are **not redistributed** here unless their licenses explicitly permit it. `data/governance/public_source_statistics.csv` stores the extracted statistics, source URLs, and access date needed to audit the matrix. For long-term preservation, add legally permitted source snapshots or archived URLs before release.

## Citation and DOI

Citation metadata are provided in `CITATION.cff`. Version 1.0.0 is archived at Zenodo under DOI [`10.5281/zenodo.22774719`](https://doi.org/10.5281/zenodo.22774719).

## License

- Source code in `src/` is licensed under the MIT License.
- Author-created data, results, figures, and documentation are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

See `LICENSE.md` and `LICENSES/` for the applicable terms.
