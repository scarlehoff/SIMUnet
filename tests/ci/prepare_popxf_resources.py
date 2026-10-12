"""Create an isolated CI resource profile and fetch the runcard's public inputs."""
import os
from pathlib import Path

from validphys.utils import yaml_safe

cache = Path(os.environ["SIMUNET_TEST_CACHE"]).resolve()
cache.mkdir(parents=True, exist_ok=True)
profile_path = Path(os.environ["NNPDF_PROFILE_PATH"])
profile = {
    "nnpdf_share": str(cache / "NNPDF"),
    "theory_urls": ["https://nnpdf.nikhef.nl/nnpdf/theories/"],
}
with profile_path.open("w") as stream:
    yaml_safe.dump(profile, stream)

# Profile must exist before constructing any loader.
from validphys.loader import FallbackLoader
import lhapdf

# validphys downloads into the last writable LHAPDF search path. Keep that
# destination inside the Actions cache rather than inside the conda environment.
pdf_cache = cache / "lhapdf"
pdf_cache.mkdir(exist_ok=True)
lhapdf.setPaths([*lhapdf.paths(), str(pdf_cache)])

card_path = Path(__file__).resolve().parents[2] / "runcards/simufac_to_popxf.yaml"
card = yaml_safe.load(card_path.read_text())
pdf = card["pdf"]
pdf_id = pdf["id"] if isinstance(pdf, dict) else pdf
loader = FallbackLoader()
loader.check_pdf(pdf_id)
loader.check_theoryID(card["theoryid"])
print(f"{pdf_id} and theory {card['theoryid']} are available.")
