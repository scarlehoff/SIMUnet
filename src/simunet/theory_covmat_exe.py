#!/usr/bin/env python
"""Build a theory covariance matrix from PineAPPL grids.

This script follows closely the setup in the ``pineappl_example.ipynb`` found in the NNPDF repo
being the main differences that:
    - It can be applied to a set of data (from some runcard)
    - It also does alpha_s variations
    - It is a script and not a notebook

The covariance matrix is then saved as a ``csv`` file.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import functools
from pathlib import Path

from lhapdf import setVerbosity
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pineappl

from nnpdf_data.utils import yaml_safe
from validphys.api import API
from validphys.convolution import OP
from validphys.pineparser import EXT
from validphys.theorycovariance.construction import (
    covs_pt_prescrip_mhou as validphys_covs_pt_prescrip,
)

setVerbosity(0)

# Since the _actual_ theory ID is (should?) be irrelevant, just put some number that
# you have already downloaded. This will be used to organize the scale-varied result later.
tid = 41000000

# Passing these configurations through validphys would make the monkeypatching more complicated
# but that would be the right way ^^U
CONFIG_HOLDER = {
    "grid_path": Path("./grid_files"),
    "scales": [(1.0, 1.0, 0.0)],
    "apply_bin_normalization": False,
    # This is to be used for DIS grids... which potentially will create inconsistencies in the
    # evolution of DIS data (ekos from the fktable) and the hadronic data (PDFs from LHAPDF)
    "alphas_pdfs": None,
}

# Some defaults:
DEFAULT_PDF = "NNPDF40MC_nnlo_as_01180"
DEFAULT_ALPHAS_PDF_LOWER = "NNPDF40_nnlo_as_01160"
DEFAULT_ALPHAS_PDF_UPPER = "NNPDF40_nnlo_as_01200"
DEFAULT_POINT_PRESCRIPTION = "7 point"

POINT_PRESCRIPTION_SCALES = {
    "3 point": [(1.0, 1.0, 0.0), (2.0, 2.0, 0.0), (0.5, 0.5, 0.0)],
    "7 point": [
        (1.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
        (0.5, 1.0, 0.0),
        (1.0, 2.0, 0.0),
        (1.0, 0.5, 0.0),
        (2.0, 2.0, 0.0),
        (0.5, 0.5, 0.0),
    ],
    "9 point": [
        (1.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
        (0.5, 1.0, 0.0),
        (1.0, 2.0, 0.0),
        (1.0, 0.5, 0.0),
        (2.0, 2.0, 0.0),
        (0.5, 0.5, 0.0),
        (2.0, 0.5, 0.0),
        (0.5, 2.0, 0.0),
    ],
}

POINT_PRESCRIPTION_ALIASES = {"3": "3 point", "7": "7 point", "9": "9 point"}


class PineObject:

    def __init__(self, pine_path, factor=1.0, shift=0, cfac=None):
        """Load one PineAPPL grid and store its dataset modifiers."""
        self._grid = pineappl.grid.Grid.read(pine_path)

        # Not all grids need normalization???
        # set it to true _for the time being_
        self._apply_bin = CONFIG_HOLDER["apply_bin_normalization"]

        self._factor = factor
        self._name = pine_path.name
        self._shift = shift
        self._cfac = cfac

    @functools.lru_cache
    def convolute(self, pdf):
        """Convolute the grid with a PDF as (nmembers, ndata, nscales)."""
        if not hasattr(pdf, "members"):
            pdf = pdf.load()
        ret = []
        bin_norm = self._grid.bin_normalizations().reshape(-1, 1)

        # Take only the QCD corerctions
        all_ord = self._grid.orders()
        mask_nnlo = pineappl.boc.Order.create_mask(all_ord, 3, 0, False)
        scales = CONFIG_HOLDER["scales"]

        for _, member in enumerate(pdf.members):
            tmp = self._grid.convolve(
                pdg_convs=self._grid.convolutions,
                xfxs=[member.xfxQ2],
                alphas=member.alphasQ2,
                order_mask=mask_nnlo,
                xi=scales,
            ).reshape(-1, len(scales))

            if self._apply_bin:
                tmp *= bin_norm

            if self._cfac is not None:
                tmp *= self._cfac[:, np.newaxis]

            # Apply shifts (if any) usually 0:
            tmp = np.concatenate([np.zeros((self._shift, len(scales))), tmp])

            ret.append(tmp)

            # keeping the loop from the notebook just in case, but we only want the first member
            break

        return np.array(ret) * self._factor

    def __str__(self):
        """Return a compact grid object label."""
        return f"PineObject({self._name})"

    def __repr__(self):
        """Return the printable grid object representation."""
        return str(self)


class PineContainer:

    def __init__(self, pine_objects, dsname=None, operation=OP["NULL"]):
        """Store grid objects and the validphys operation that combines them."""
        self._name = dsname
        self._operation = operation
        self._pine_objects = pine_objects

    @functools.lru_cache
    def predictions(self, pdf):
        """Compute predictions for all contained grids and scales."""
        operators = []
        for pine_operator in self._pine_objects:
            tmp = []
            for pine_bin in pine_operator:
                tmp.append(pine_bin.convolute(pdf))
            # tmp is shaped (ndata, scales)
            operators.append(np.concatenate(tmp, axis=1))

        # The operators is a list of (nmembers, ndata, nscales)

        # Loop over scales to get the result for all members for every scale
        return self._operation(*operators)  # (nmembers, ndata, nscales)

    def __str__(self):
        """Return a compact container label."""
        return f"PineContainer({self._name})"

    def __repr__(self):
        """Return the printable container representation."""
        return str(self)


def _resolve_pine_path(grid_name):
    """Find the grid file in the grid directory.
    Optionally, use a FkTable path. This will be necessary for DIS."""
    pine_path = CONFIG_HOLDER["grid_path"] / f"{grid_name}.{EXT}"
    if pine_path.exists():
        return pine_path

    raise FileNotFoundError(f"Could not find grid for {grid_name}: expected {pine_path}")


# This block contains the whole monkeypatching logic.
import validphys.results
import validphys.theorycovariance.construction


@functools.lru_cache
def _get_pine_container(dataset):
    """Build a cached PineContainer for a validphys dataset."""
    cd = dataset.commondata
    metadata = cd.metadata
    theory_meta = metadata.theory
    fk_specs = dataset.fkspecs

    print(f"Normalization factors: {theory_meta.conversion_factor}")

    pinegrids = []
    for operator, fk_spec in zip(theory_meta.FK_tables, fk_specs):
        tmp = []
        cfac = None
        for i in operator:
            factor = theory_meta.conversion_factor
            shift = 0
            if theory_meta.normalization is not None:
                print(f"Normalization factor for {i}: {theory_meta.normalization.get(i, None)}")
                factor *= theory_meta.normalization.get(i, 1.0)

            if theory_meta.shifts is not None:
                shift = theory_meta.shifts.get(i, 0)

            if len(fk_spec.load_cfactors()) > 0:
                cfac_data = fk_spec.load_cfactors()
                if len(cfac_data) > 1:
                    raise Warning(
                        f"TODO: only one cfactor can be used, more than one not implemented."
                    )
                cfac = np.concatenate([i.central_value for i in cfac_data[0]])

            pine_path = CONFIG_HOLDER["grid_path"] / f"{i}.{EXT}"
            tmp.append(PineObject(pine_path, factor, shift=shift, cfac=cfac))
        pinegrids.append(tmp)

    operation = OP[theory_meta.operation]
    return PineContainer(pinegrids, dsname=dataset.name, operation=operation)


def _pine_predictions(dataset, pdf, central_only=False):
    """Given a dataset and a PDF, produces predictions with pineappl
    The output shape is a list of DataFrames with the right shape for ThPredictions"""
    if central_only:
        pdf = pdf.load_t0()

    container = _get_pine_container(dataset)
    res_all_scales = container.predictions(pdf)  # (n_members, n_data, n_scales)
    cuts = dataset.cuts.load() if dataset.cuts else None

    all_res = []
    for res in res_all_scales.T:
        all_res.append(pd.DataFrame(res).loc[cuts])

    return all_res


def new_results_central_by_theoryid(dataset, pdf, covariance_matrix, sqrt_covmat):
    """Return central predictions tagged by synthetic theory IDs."""
    dresult = validphys.results.DataResult(dataset, covariance_matrix, sqrt_covmat)
    ####### This is the part that changes wrt validphys
    # and, with respect to the notebook, we have added the alpha_s path
    ret = []
    alphas_pdfs = CONFIG_HOLDER["alphas_pdfs"]
    if alphas_pdfs is None:
        theory_data = _pine_predictions(dataset, pdf, central_only=True)
        theory_pdfs = [pdf] * len(theory_data)
    else:
        theory_data = [
            _pine_predictions(dataset, alphas_pdf, central_only=True)[0]
            for alphas_pdf in alphas_pdfs
        ]
        theory_pdfs = alphas_pdfs
    for i, (data, theory_pdf) in enumerate(zip(theory_data, theory_pdfs)):
        tmp = validphys.results.ThPredictionsResult(
            data, theory_pdf.stats_class, pdf=theory_pdf, theoryid=tid + i
        )
        ret.append((dresult, tmp))
    #########
    return ret


def new_results(dataset, pdf, covariance_matrix, sqrt_covmat, central_only=False):
    """Return data and theory results using PineAPPL predictions."""
    dresult = validphys.results.DataResult(dataset, covariance_matrix, sqrt_covmat)
    ####### This is the part that changes wrt validphys
    theory_data = _pine_predictions(dataset, pdf, central_only=central_only)[0]
    theory_results = validphys.results.ThPredictionsResult(theory_data, pdf.stats_class, pdf=pdf)
    #########
    return (dresult, theory_results)


def new_results_central(dataset, pdf, covariance_matrix, sqrt_covmat, central_only=True):
    """Return central data and theory results using PineAPPL predictions."""
    return new_results(dataset, pdf, covariance_matrix, sqrt_covmat, central_only=central_only)


def apply_pineappl_monkeypatch():
    """Install the validphys monkeypatches."""
    validphys.results.results = new_results
    validphys.results.results_central = new_results_central
    validphys.theorycovariance.construction.results_central_bytheoryids = (
        new_results_central_by_theoryid
    )
    validphys.theorycovariance.construction.covs_pt_prescrip_mhou = new_covs_pt_prescrip

    # Make sure that from_convolution is never accessed
    def raise_me(pdf, dataset, **kwargs):
        """Reject accidental use of the original convolution path."""
        raise ValueError(".from_convolution is being used, please report this error!")

    validphys.results.ThPredictionsResult.from_convolution = raise_me


@dataclass
class DatasetPredictions:
    """Store predictions for one dataset and its process group."""

    name: str
    process: str
    predictions: NDArray[np.float64]


def parse_args():
    """Parse command-line options for covariance generation."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Example: python grid_theory_covmat.py example.yaml --point-prescription "7 point"',
    )
    parser.add_argument(
        "runcard", type=Path, help="Runcard containing dataset_inputs and theory::theoryid"
    )
    parser.add_argument(
        "--point-prescription",
        default=DEFAULT_POINT_PRESCRIPTION,
        help=(
            'Point prescription: "3 point", "7 point", "9 point" '
            f"(or 3, 7, 9; default: {DEFAULT_POINT_PRESCRIPTION})"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("datacuts_theory_theorycovmatconfig_theory_covmat_custom.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--grid-dir", type=Path, default=Path("grid_files"), help="Directory with PineAPPL grids"
    )
    parser.add_argument(
        "--pdf",
        default=DEFAULT_PDF,
        help=(
            "PDF to use when the runcard has no theorycovmatconfig::pdf "
            f"(default: {DEFAULT_PDF})"
        ),
    )
    parser.add_argument(
        "--alphas-variation",
        action="store_true",
        help="Build an alpha_s covariance matrix instead of a scale-variation covariance matrix",
    )
    parser.add_argument(
        "--alphas-pdf-lower",
        default=DEFAULT_ALPHAS_PDF_LOWER,
        help=f"Lower-alpha_s PDF for --alphas-variation (default: {DEFAULT_ALPHAS_PDF_LOWER})",
    )
    parser.add_argument(
        "--alphas-pdf-upper",
        default=DEFAULT_ALPHAS_PDF_UPPER,
        help=f"Upper-alpha_s PDF for --alphas-variation (default: {DEFAULT_ALPHAS_PDF_UPPER})",
    )
    parser.add_argument(
        "--strict-grid-dir",
        action="store_true",
        help="Do not fall back to the theory FK-table path if a grid is absent from --grid-dir",
    )
    parser.add_argument(
        "--apply-bin-normalization",
        action="store_true",
        help="Apply PineAPPL bin normalizations, matching the notebook's APPLY_BIN_NORMALIZATION switch",
    )
    return parser.parse_args()


def normalize_point_prescription(point_prescription):
    """Accept 3, 7 9 and N point."""
    normalized = " ".join(point_prescription.lower().split())
    if normalized in POINT_PRESCRIPTION_SCALES:
        return normalized
    if normalized in POINT_PRESCRIPTION_ALIASES:
        return POINT_PRESCRIPTION_ALIASES[normalized]
    valid = ", ".join((*POINT_PRESCRIPTION_SCALES, *POINT_PRESCRIPTION_ALIASES))
    raise ValueError(
        f"Unsupported point prescription {point_prescription!r}. "
        f"These are the options available: {valid}"
    )


def base_validphys_config(runcard, pdf_name="NNPDF40MC_nnlo_as_01180"):
    """Build the validphys configuration passed to API calls.
    Assuming a n3fit-like runcard.
    If it isn't / we want something else, this should be updated.
    While the PDF is not supposed to be in the runcard, it won't be overwritten if it is.
    """
    config = dict(runcard)
    config["dataset_inputs"] = runcard["dataset_inputs"]
    config["theoryid"] = runcard["theory"]["theoryid"]
    config.setdefault("pdf", pdf_name)
    config.setdefault("use_cuts", "internal")
    return config


def new_covs_pt_prescrip(combine_by_type, point_prescription):
    """Call validphys block construction without reportengine checks."""
    return validphys_covs_pt_prescrip(combine_by_type, point_prescription)


def theory_covmat_for_prescription(config, point_prescription):
    """Build a scale-variation theory covariance matrix.
    If point_prescription is not in POINT_PRESCRIPTION_SCALES it will
    fail non graciously but in principle this has already been checked at this point.
    """
    CONFIG_HOLDER["alphas_pdfs"] = None
    CONFIG_HOLDER["scales"] = POINT_PRESCRIPTION_SCALES[point_prescription]
    _get_pine_container.cache_clear()

    print(
        f"Building {point_prescription} covariance with {len(CONFIG_HOLDER['scales'])} scale points"
    )
    return API.theory_covmat_custom_per_prescription(
        point_prescription=point_prescription, **config
    )


def theory_covmat_for_alphas(config, pdfs):
    """Build an alpha_s theory covariance matrix.
    This function sets the scales to (1,1,0) and tries to recover the previous
    value at the end.
    """
    original_value = deepcopy(CONFIG_HOLDER["scales"])
    CONFIG_HOLDER["scales"] = [(1.0, 1.0, 0.0)]
    CONFIG_HOLDER["alphas_pdfs"] = pdfs
    _get_pine_container.cache_clear()

    print("Building alpha_s covariance with central, lower, and upper PDFs")
    try:
        return API.theory_covmat_custom_per_prescription(point_prescription="alphas", **config)
    finally:
        CONFIG_HOLDER["scales"] = original_value
        CONFIG_HOLDER["alphas_pdfs"] = None


def main():
    """Run the command-line covariance generation workflow."""
    args = parse_args()
    runcard = yaml_safe.load(args.runcard.read_text())

    CONFIG_HOLDER["grid_path"] = args.grid_dir
    CONFIG_HOLDER["apply_bin_normalization"] = args.apply_bin_normalization

    pdf_name = args.pdf
    if args.pdf is None:
        # Try to get it from the runcard and fail non-graciously otherwise
        pdf_name = runcard["theorycovmatconfig"]["pdf"]

    config = base_validphys_config(runcard, pdf_name)

    apply_pineappl_monkeypatch()

    if args.alphas_variation:
        alphas_pdfs = [
            API.pdf(pdf=pdf_name),
            API.pdf(pdf=args.alphas_pdf_lower),
            API.pdf(pdf=args.alphas_pdf_upper),
        ]
        theory_covmat = theory_covmat_for_alphas(config, alphas_pdfs)
    else:
        try:
            point_prescription = normalize_point_prescription(args.point_prescription)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        theory_covmat = theory_covmat_for_prescription(config, point_prescription)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Saves the theory covmat as a table a la reportengine
    theory_covmat.to_csv(args.output, sep='\t', na_rep='nan')
    print(
        f"Wrote {theory_covmat.shape[0]}x{theory_covmat.shape[1]} theory covariance matrix to {args.output}"
    )


if __name__ == "__main__":
    main()
