# Covariance matrices in the n3fit step (SIMUnet/NNPDF)

This note explains **where the covariance matrices used by `n3fit` come from** and **how they are constructed** in this codebase.

## TL;DR

- `n3fit` itself does **not** implement the experimental covariance-matrix construction.
- Covariance matrices are produced by the **`validphys` provider graph** (ReportEngine-style) using the dataset **CommonData** (stat + syst tables).
- The key implementation is in:
  - `validphys2/src/validphys/covmats.py` (Python construction, T0 option, weights, optional theory-covmat addition, Cholesky square-root)
  - `libnnpdf/src/chisquared.cc` (C++ implementation of the same prescription; useful for cross-checks)

## Where `n3fit` plugs into this

`vp-setupfit` is essentially a “validphys app wrapper” that prepares the fit folder and runs a fixed set of actions/providers.

- Providers include `validphys.covmats` and `validphys.theorycovariance.construction` (see `n3fit/src/n3fit/scripts/vp_setupfit.py`).
- If your runcard contains `theorycovmatconfig`, `vp-setupfit` will schedule the action `nnfit_theory_covmat` (also in `vp_setupfit.py`).

From the point of view of n3fit training, what matters is that the runcard evaluation produces a:

- full covariance matrix (or equivalently its inverse / square-root depending on the code path)
- consistent ordering with the post-cut datapoint ordering

## Inputs: what the covariance matrix is built from

The covariance matrix is constructed from the dataset’s CommonData content:

- statistical uncertainties (always contribute to the diagonal)
- systematic uncertainties, with per-systematic metadata describing:
  - whether the systematic is additive (ADD) or multiplicative (MULT)
  - whether it is correlated or uncorrelated (and “theory” variants)
  - a special convention: if the systematic name is `SKIP`, it is ignored

The CommonData objects are produced earlier in the pipeline (filtering/cuts), but the covariance logic itself lives in `validphys`.

## Python implementation: `validphys2/src/validphys/covmats.py`

### 1) Single-dataset covariance matrix

`covmat_from_systematics(...)` builds the dataset covariance matrix by calling `construct_covmat(stat_errors, systematic_errors)`.

Key behavior:

- ADD systematics are already in “absolute units” and contribute directly.
- MULT systematics are percentages and are converted to absolute shifts via:

  $$\Delta_i^{\mathrm{abs}} = \Delta_i^{\%} \times \mathrm{central}_i / 100$$

- The total covariance matrix has the schematic structure:

  $$C = \mathrm{diag}(\sigma_\mathrm{stat}^2 + \sigma_\mathrm{uncorr}^2) + A_{\mathrm{corr}} A_{\mathrm{corr}}^T$$

### 2) Multiple datasets (dataset_inputs) + cross-dataset correlations

`dataset_inputs_covmat_from_systematics(...)` generalizes this to a list of datasets.

Important detail: **SPECIAL correlated** systematics can correlate *across datasets*.

- Intra-dataset systematics (`UNCORR`, `CORR`, `THEORYUNCORR`, `THEORYCORR`) only populate the block diagonal.
- “Special correlated” systematics are concatenated across all datasets and then contribute as an outer product, producing off-block-diagonal terms.

### 3) Dataset weights

If `use_weights_in_covmat=True` (the default in these helpers):

- For a single dataset, `covmat /= weight`.
- For dataset_inputs, it applies the pairwise scaling

  $$C_{ij} \to \frac{C_{ij}}{\sqrt{w_i}\sqrt{w_j}}$$

where $w_i$ is the dataset weight of the datapoint that index $i$ belongs to.

### 4) The T0 option (`use_t0`)

`n3fit` fits typically use the **T0 prescription** for multiplicative errors.

Why: multiplicative uncertainties depend on the choice of “central values” used to convert % errors to absolute units.

- With `use_t0: false` the conversion uses the experimental central values.
- With `use_t0: true` the conversion uses **theory predictions** (the “t0 predictions”) from the central member of a chosen `t0pdfset`.

The T0 path is implemented via:

- `dataset_t0_predictions(...)` (uses `validphys.convolution.central_predictions`)
- `t0_covmat_from_systematics(...)`
- `dataset_inputs_t0_covmat_from_systematics(...)`

Selection of which implementation is used is done in `validphys2/src/validphys/config.py`:

- `produce_covmat_t0_considered(use_t0=...)`
- `produce_dataset_inputs_covmat_t0_considered(use_t0=...)`

### 5) Optional: add theory covariance matrix (exp + th)

If the fit provides a theory covariance matrix (via `theorycovmatconfig` and `nnfit_theory_covmat`), validphys can add it to the experimental covmat:

- `reorder_thcovmat_as_expcovmat(...)` aligns ordering to the experimental covmat order.
- `_dataset_inputs_covmat_t0_considered(...)` adds the reordered theory covmat values if present.

### 6) Square-root covariance matrix (Cholesky)

`dataset_inputs_sqrt_covmat(...)` calls `sqrt_covmat(...)`.

The Cholesky is done in a numerically-stable way:

- build correlation matrix $\rho$ from $C$
- compute Cholesky of $\rho$
- rescale back to $\sqrt{C}$

This is specifically meant to behave better when the covariance matrix is close to singular.

## C++ implementation (cross-check): `libnnpdf/src/chisquared.cc`

The same conceptual model exists in C++ and is used by the native fitting code:

- `ComputeCovMat_basic(...)`:
  - adds stat errors on the diagonal
  - adds correlated systematics as products
  - converts multiplicative systematics using central values (`central_values[i]*central_values[j]*1e-4` corresponds to `/100` twice)
  - ignores systematics named `SKIP`
  - optionally adds a theory covariance matrix from file (`read_theory_covmat`)
  - applies dataset weights via division by `sqrt_weights[i]*sqrt_weights[j]`
- `ComputeSqrtMat(...)` implements the “Cholesky via correlation matrix” strategy, matching the Python `sqrt_covmat(...)` approach.

## Practical navigation: “what to open when debugging covmats”

- Main Python implementation: `validphys2/src/validphys/covmats.py`
- Dispatch / configuration knobs (`use_t0`, `use_pdferr`): `validphys2/src/validphys/config.py`
- Setupfit wiring (providers + theorycovmatconfig hook): `n3fit/src/n3fit/scripts/vp_setupfit.py`
- Native reference implementation: `libnnpdf/src/chisquared.cc`

## Common gotchas

- If `use_t0: true`, you must also provide a valid `t0pdfset` (config enforces this).
- If a theory covariance matrix is provided, it must match the cut dataset ordering; validphys includes checks and an explicit reorder helper.
- Near-singular covmats are handled by the correlation-matrix Cholesky strategy; if you see decomposition failures, inspect whether regularisation (`norm_threshold`) is used in your action path.
