"""Get Slices

Reads the annotation CSV and extracts the specified slices
The slices are saved as PNG images with the slice ID in the file name

Usage:
  get_slices.py <sample>
  get_slices.py (-h | --help)

Options:
  -h --help         Show this screen

"""
import os
import h5py
import pandas as pd
from skimage import io
from docopt import docopt

from .. import config


def main():
    args = docopt(__doc__)
    sample = args["<sample>"]
    annotation_file = os.path.join(config.ANNOTATION_CSV, f"{sample}.csv")

    if not os.path.exists(annotation_file):
        raise RuntimeError(f"No annotations for {sample} were found")

    dst_dir = os.path.join(config.IMG_SLICES_DIR, sample)
    os.makedirs(dst_dir, exist_ok=True)
    df = pd.read_csv(annotation_file).drop(columns=["z_min", "z_max"])
    
    for row in df.itertuples():
        tomo_name = row[1]
        tomo_path = os.path.join(config.RAW_TOMO_DIR, sample, tomo_name)
        
        with h5py.File(tomo_path) as fh:
            for s in row[2:]:
                out_path = os.path.join(dst_dir, f"{tomo_name[:-4]}_{s}.png")
                io.imsave(out_path, fh["data"][s])


if __name__ == '__main__':
    main()
