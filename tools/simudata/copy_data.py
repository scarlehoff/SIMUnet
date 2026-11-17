"""
This script copies SIMU factor files from the source directory into the simunet package's data directory based on a mapping provided in `data_maps.csv`. 
It checks for the existence of the source files and only copies those that are valid according to specified criteria. Any skipped entries are recorded in `data_maps2.csv`.
"""
import os
import shutil
import csv
from pathlib import Path

# Paths
simu_factors_dir = os.path.expanduser(
    "~/miniconda3/envs/simunet_mac/share/NNPDF/data/theory_270/simu_factors"
)
target_dir = Path(__file__).resolve().parents[2] / "src/simunet/simudata/simu_factors"
csv_file_path = Path(__file__).with_name("data_maps.csv")
output_csv = Path(__file__).with_name("data_maps2.csv")


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def is_valid(row):
    new_name = row['new_name'].strip()
    simu_fac = (row.get('simu_fac') or '').strip()

    return (
        new_name != "NOT FOUND"
        and "EFT" in simu_fac
    )


def process(datasets):
    skipped = []

    for row in datasets:
        old_name = row['old_name'].strip()
        new_name = row['new_name'].strip()
        simu_fac = (row.get('simu_fac') or '').strip()

        simu_factor_file = f"SIMU_{old_name}.yaml"

        if is_valid(row):
            print(f"Processing {old_name} -> {new_name} [{simu_fac}]")

            source_file = os.path.join(simu_factors_dir, simu_factor_file)

            if os.path.exists(source_file):
                dest_file = os.path.join(
                    target_dir, f"SIMU_{new_name}.yaml"
                )
                shutil.copy(source_file, dest_file)
                print(f"Copied → {dest_file}")
            else:
                print(f"Missing file: {source_file}")
                skipped.append(row)
        else:
            print(f"Skipping {old_name}")
            skipped.append(row)

    return skipped


def write_skipped(skipped):
    with open(output_csv, "w", newline='') as f:
        fieldnames = ["old_name", "new_name", "cfac", "simu_fac"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(skipped)


if __name__ == "__main__":
    datasets = read_csv(csv_file_path)
    skipped = process(datasets)
    write_skipped(skipped)
