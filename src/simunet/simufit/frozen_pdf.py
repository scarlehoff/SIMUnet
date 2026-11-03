"""Export frozen LHAPDF fits in the format expected by NNPDF postfit."""

import functools
import inspect
import json
import re

import numpy as np

from n3fit.vpinterface import compute_arclength, integrability_numbers
from validphys.core import PDF
from validphys.lhapdf_compatibility import make_pdf
from validphys.lhapdfset import LHAPDFSet
from validphys.utils import yaml_safe


class _LHAPDFMember(LHAPDFSet):
    """Use Validphys's grid evaluation with only the requested LHAPDF member."""

    def __init__(self, name, member):
        self._name = name
        # Validphys's t0 mode means one active member, supplied explicitly here.
        self._error_type = "t0"
        self._lhapdf_set = make_pdf(name, member)
        self._flavors = None


class FrozenPDFMember(PDF):
    """A single Monte Carlo replica exposed through the Validphys PDF interface.

    Both ordinary and central convolutions evaluate this member. Include its
    index in the cache key so predictions for different replicas stay distinct.
    """

    def __init__(self, pdf, member):
        if pdf.error_type != "replicas":
            raise ValueError("frozen_pdf must be a Monte Carlo LHAPDF set (ErrorType: replicas)")
        if not isinstance(member, (int, np.integer)) or not 1 <= member < len(pdf):
            raise ValueError(
                f"{pdf.name}: replica {member} is unavailable; expected a member between "
                f"1 and {len(pdf) - 1}"
            )
        self.member = int(member)
        super().__init__(pdf.name)
        self.comp_tuple = (pdf.name, self.member)
        if not self.member_path.is_file():
            raise ValueError(f"{pdf.name}: LHAPDF replica {member} is missing: {self.member_path}")

    @property
    def member_path(self):
        return self.infopath.with_name(f"{self.name}_{self.member:04d}.dat")

    @functools.lru_cache(maxsize=16)
    def load(self):
        return _LHAPDFMember(self.name, self.member)

    def load_t0(self):
        "Load the single corresponding LHAPDF member"
        return self.load()

    def __len__(self):
        return 1

    def make_only_cv(self):
        return self


def write_frozen_pdf_fit(writer, save_path, fitname, frozen_pdf):
    """Write diagnostics and the matching frozen replica.

    Preserve the frozen PDF's full grid and evolution metadata.
    """
    if tuple(writer.replica_numbers) != (frozen_pdf.member,):
        raise ValueError("The exported frozen PDF member must match the fitted replica")
    save_path.mkdir(parents=True, exist_ok=True)
    arc_lengths = compute_arclength(frozen_pdf).tolist()
    integrability = integrability_numbers(frozen_pdf).tolist()
    if not np.isfinite(arc_lengths).all() or not np.isfinite(integrability).all():
        raise ValueError(f"{frozen_pdf.name}: non-finite frozen PDF diagnostics")

    source = frozen_pdf.member_path
    member = re.sub(r"(?m)^PdfType:.*$", "PdfType: replica", source.read_text(), count=1)
    info = dict(frozen_pdf.info)
    info.update(
        SetDesc=f"BSM fit {fitname} with frozen replicas from {frozen_pdf.name}",
        NumMembers="REPLACE_NREP",
        ErrorType="replicas",
    )
    # The new ensemble has its own identity and Monte Carlo error convention.
    info.pop("SetIndex", None)
    info.pop("ErrorConfLevel", None)
    with (save_path / f"{fitname}.info").open("w") as stream:
        yaml_safe.dump(info, stream)

    count = len(writer.replica_numbers)
    writer.preprocessing = [None] * count  # No fitted PDF preprocessing exponents.
    writer.arc_lengths = [arc_lengths] * count
    writer.integrability_numbers = [integrability] * count
    metadata_has_replica = (
        "replica_number" in inspect.signature(writer._write_metadata_json).parameters
    )
    provenance = {"pdf": frozen_pdf.name, "member": frozen_pdf.member}
    for i, replica in enumerate(writer.replica_numbers):
        replica_path = save_path / f"replica_{replica}"
        replica_path.mkdir(parents=True, exist_ok=True)
        writer._write_chi2s(replica_path / "chi2exps.log")
        metadata_path = replica_path / f"{fitname}.json"
        metadata_kwargs = {"replica_number": replica} if metadata_has_replica else {}
        writer._write_metadata_json(i, out_path=metadata_path, **metadata_kwargs)
        metadata = json.loads(metadata_path.read_text())
        metadata["frozen_pdf"] = provenance
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        (replica_path / f"{fitname}.dat").write_text(member)
        (replica_path / "frozen_pdf.json").write_text(json.dumps(provenance, indent=2) + "\n")
