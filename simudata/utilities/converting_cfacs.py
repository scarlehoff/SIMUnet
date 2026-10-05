
import csv
import logging
import shutil
from pathlib import Path
from typing import List, Tuple
from yaml import safe_load, YAMLError
from conversor import _autoname

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CSV_FILE_PATH = Path("data_maps.csv")
TARGET_DIR = Path("/Users/ellacole/codes/simunet/simunet_nnpdf/simudata/commondata")
CFACTORS_DIR = Path("/Users/ellacole/miniconda3/envs/simunet_mac/share/NNPDF/data/theory_270/cfactor")
NEW_CFAC_DIR = TARGET_DIR / "cfactors"

def _parse_cfactor_names(raw_value: str) -> List[str]:
    """Parses and cleans cfactor names from a raw string format."""
    cleaned = raw_value.strip("[]").replace("'", "")
    return cleaned.split()

def _process_metadata(target_path: Path, col1: str) -> Tuple[List[str], List[str]]:
    """
    Extracts relevant FK tables and ends from the dataset's metadata file.
    
    Returns:
        A tuple of (fk_names, ends)
    """
    metadata_file = target_path / "metadata.yaml"
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = safe_load(f)

    observables = metadata.get("implemented_observables", [])
    if not observables:
        raise ValueError(f"No implemented observables found in metadata for {target_path.name}")

    if len(observables) > 1:
        obs_name = col1.split("_")[-1]
        obs = next(
            (item for item in observables if item.get("observable_name") == obs_name), 
            None
        )
        if not obs:
            raise ValueError(f"Observable '{obs_name}' not found in metadata for {target_path.name}")
    else:
        obs = observables[0]

    fk_tables = obs.get("theory", {}).get("FK_tables", [])
    fk_names = [table[0] if isinstance(table, list) else table for table in fk_tables]

    if len(fk_names) > 1:
        ends = [f"_{fk_name.split('_')[-1]}" for fk_name in fk_names]
    else:
        ends = [""]

    return fk_names, ends

def _copy_cfactor_files(col1: str, cfactor_names: List[str], ends: List[str], fk_names: List[str]) -> None:
    """Handles renaming and copying of the old cfactor files to the new directory."""
    NEW_CFAC_DIR.mkdir(parents=True, exist_ok=True)

    for cfac_type in cfactor_names:
        for idx, end in enumerate(ends):
            old_cfac_name = f"CF_{cfac_type}_{col1}{end}.dat"
            old_cfac_path = CFACTORS_DIR / old_cfac_name
            
            if not old_cfac_path.exists():
                logger.warning(f"Source cfactor file missing: {old_cfac_path}")
                continue

            try:
                fk_name = fk_names[idx]
            except IndexError:
                logger.error(f"Index mismatch between FK tables and file ending segments at index {idx}.")
                continue

            new_cfac_name = f"CF_{cfac_type}_{fk_name}.dat"
            new_cfac_path = NEW_CFAC_DIR / new_cfac_name

            try:
                shutil.copy(old_cfac_path, new_cfac_path)
                logger.info(f"Successfully copied '{old_cfac_name}' -> '{new_cfac_name}'")
            except IOError as e:
                logger.error(f"Failed to copy '{old_cfac_name}': {e}")

def migrate_cfactors(csv_path: Path, search_base_dir: Path) -> None:
    """
    Scans the data maps CSV file for unmapped data entries, maps names 
    via autoname, and reorganizes corresponding cfactor files.
    """
    if not search_base_dir.exists():
        logger.error(f"Target directory does not exist: {search_base_dir}")
        return

    logger.info(f"Scanning {csv_path.name} for 'NOT FOUND' records...")
    logger.info(f"Target Base Directory: {search_base_dir}")

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f, skipinitialspace=True)
            next(reader, None) 

            for row_idx, row in enumerate(reader, start=2):
                if not row or len(row) < 2:
                    continue

                col1 = row[0].strip()
                col2 = row[1].strip()

                col3_has_value = len(row) >= 3 and row[2].strip() != ""
                if not col3_has_value:
                    continue

                if col2.upper() in ["NOT FOUND", "NOT_FOUND"]:
                    logger.info(f"Processing Row {row_idx}: Mapping legacy name '{col1}'")

                    try:
                        generated_new_name = _autoname(col1)
                        set_folder_name = generated_new_name.rsplit("_", 1)[0]
                        target_path = search_base_dir / set_folder_name

                        if not target_path.is_dir():
                            logger.error(f"Expected directory path '{target_path}' does not exist.")
                            continue

                        cfactor_names = _parse_cfactor_names(row[2])
                        fk_names, ends = _process_metadata(target_path, col1)
                        
                        _copy_cfactor_files(col1, cfactor_names, ends, fk_names)

                    except NotImplementedError:
                        logger.warning(f"Row {row_idx}: Skipped - Prefix unrecognized by _autoname.")
                    except (YAMLError, ValueError, FileNotFoundError) as e:
                        logger.error(f"Row {row_idx}: Failed to process metadata for '{col1}': {e}")

    except FileNotFoundError:
        logger.error(f"The tracking CSV file was not found at: {csv_path}")

if __name__ == "__main__":
    migrate_cfactors(CSV_FILE_PATH, TARGET_DIR)
