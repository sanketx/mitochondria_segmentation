"""Granule Masks

Extracts granule annotations from the old 3D UNet predictions
Annotation file is used to extract slices of granules
The mito mask is used to remove granules outside mitochondria
The slices are saved as PNG images with the slice ID in the file name

Usage:
  granule_masks.py <sample>
  granule_masks.py (-h | --help)

Options:
  -h --help         Show this screen

"""
import os
import h5py
import numpy as np
import pandas as pd
from skimage import io
from docopt import docopt

from .. import config

# Don't change these
THRESH = {
    "BACHD_controlRNA": 2.8,
    "Q66": 2.0,
    "Q77": 2.1,
    "BACHD": 2.5,
    "dN17_BACHD": 2.5,
    "Q18": 2.5,
    "Q53": 2.5,
    "WT": 2.5,
    "Q109": 3,
    "Q20": 0.92,
}


def extract_granules(row, sample, mito_dir, granule_dir):
    tomo_name = row[1]
    tomo_path = os.path.join(config.RAW_TOMO_DIR, sample, tomo_name)
    
    with h5py.File(tomo_path) as fh:
        for s in row[2:]:
            file_name = f"{tomo_name[:-4]}_{s}.png"
            mito_path = os.path.join(mito_dir, file_name)

            mito_mask = io.imread(mito_path)
            granule_mask = 253 * (fh["granule"][s] > THRESH[sample])

            # I messed up some rounding calcs so we are sometimes off by 1
            mito_mask = mito_mask[:granule_mask.shape[0], :granule_mask.shape[1]]
            granule_mask = np.where(mito_mask == 254, granule_mask, 0)

            out_path = os.path.join(granule_dir, file_name)
            io.imsave(out_path, granule_mask.astype(np.uint8))


def main():
    args = docopt(__doc__)
    sample = args["<sample>"]
    annotation_file = os.path.join(config.ANNOTATION_CSV, f"{sample}.csv")
    mito_dir = os.path.join(config.MITO_ANNOTATIONS, sample)
    
    if not os.path.exists(annotation_file):
        raise RuntimeError(f"No annotations for {sample} were found")

    if not os.path.exists(mito_dir):
        raise RuntimeError(f"No mitochondria masks for {sample} were found")

    granule_dir = os.path.join(config.GRANULE_ANNOTATIONS, sample)
    os.makedirs(granule_dir, exist_ok=True)
    df = pd.read_csv(annotation_file).drop(columns=["z_min", "z_max"])

    for row in df.itertuples():
        extract_granules(row, sample, mito_dir, granule_dir)


if __name__ == '__main__':
    main()
