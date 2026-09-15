# Reproducibility protocol

## Exact seeds and sample sizes

### Synthetic benchmark
- Seed: `20260909`
- Scenarios: `N = 30,000`
- TV radii: 0 to 0.30 in increments of 0.005
- Safety threshold: `tau = 0.02`
- Required safety confidence: `alpha = 0.95`
- Wasserstein surrogate: 50 stress-quantile clusters

### Public-data AI-governance application
- Seed: `20260913`
- Monte Carlo replications: `500,000`
- Criterion weights: `Dirichlet(1,1,1,1,1,1)`
- Performance within each public interval: uniform
- `g2`: one shared draw across all four alternatives per replication
- TV reporting radius: `epsilon = 0.15` (sensitivity parameter, not an empirically estimated radius)

### Contamination validation
- Frozen random-contamination dataset: 5,400 rows = 9 contamination levels × 200 replications × 3 methods.
- The archived frozen replications are the primary validation input. They reproduce the reported tables/figures exactly through `src/02_contamination_analysis.py`.

## Expected key checks

Synthetic benchmark:
- Nominal first-rank Fast + safeguards ≈ 0.660333
- Nominal first-rank Conditional scaling ≈ 0.339667
- Nominal safety Fast + safeguards ≈ 0.982400
- TV CAR(Fast, Conditional) ≈ 0.160333
- TV SAR(Fast; alpha=.95) ≈ 0.032400

AI-governance application:
- Unfiltered b1: a1=0.083778, a2=0.643706, a3=0.272486, a4=0.000030
- After safety filter {a3,a4}: a3=0.997696, a4=0.002304

## Verification

After `python src/run_all.py`, compare regenerated CSVs to archived final CSVs. Small discrepancies should not occur for the public-data main run because the exact seed/order is fixed. The contamination metrics are deterministic functions of the frozen replication file.
