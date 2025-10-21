"""
Creation of `MaldiSet` with default parameters

Usage
-----
python src/prepare_maldiset_batch.py \
    --spectra_dir </path/to/data/> \
    --metadata </path/to/metadata> \
    --antibiotics Meropenem Amikacin \
    --other City \
    --output_dir ./data/dfs/
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from maldiamrkit.dataset import MaldiSet

def main(spectra_dir, metadata_path, antibiotics, other, output_dir):
    data = MaldiSet.from_directory(
        spectra_dir,
        metadata_path,
        aggregate_by=dict(antibiotics=antibiotics, other=other),
        verbose=False
    )

    for antibiotic in antibiotics:
        try:
            fig, axes = data.plot_pseudogel(
                antibiotic=antibiotic,
                show=False,
            )
            plt.savefig(os.path.join(output_dir, f"pseudogel_{antibiotic}.png"), dpi=300)
        except Exception as e:
            print(e)

    X, y = data.X, data.y

    os.makedirs(output_dir, exist_ok=True)

    pd.concat([X, y], axis=1).to_csv(os.path.join(output_dir, "data_bin_3.csv"), index=True)
    data.other.to_csv(os.path.join(output_dir, "metadata_bin_3.csv"), index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save MALDI-TOF spectra")
    parser.add_argument("--spectra_dir", required=True, help="Folder containing the spectra")
    parser.add_argument("--metadata", required=True, help="CSV file with metadata")
    parser.add_argument("--antibiotics", nargs="+", required=True, help="The antibiotic for the MaldiSet")
    parser.add_argument("--other", required=True, help="Other metadata to include in the MaldiSet")
    parser.add_argument("--output_dir", required=True, help="Output directory for X.csv e y.csv")

    args = parser.parse_args()

    main(args.spectra_dir, args.metadata, args.antibiotics, args.other, args.output_dir)