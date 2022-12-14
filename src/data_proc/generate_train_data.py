"""Generate Train Data

Reads the annotations downloaded from hasty.ai and makes labels
The mito and granule labels are stored along with the raw_data
Z limits from the annotation file are used to annotate slices
which are confirmed to have no mito or granules

Usage:
  generate_train_data.py <sample>
  generate_train_data.py (-h | --help)

Options:
  -h --help         Show this screen

"""
import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
from scipy import ndimage
from docopt import docopt

from .. import config


def insert_labels(label, idx, mito_labels, granule_labels):
    if label.shape != mito_labels[idx].shape:  # Q109 issue
        temp_label = np.zeros(mito_labels[idx].shape, dtype=label.dtype)
        temp_label[:label.shape[0], :label.shape[1]] = label
        label = temp_label

    mito_label = np.where(label >= 253, 1, 0)
    mito_label = ndimage.binary_fill_holes(mito_label)
    mito_labels[idx] = mito_label.astype(np.int8)

    granule_label = np.where(label == 253, 1, 0)
    granule_label = ndimage.binary_fill_holes(granule_label)
    granule_labels[idx] = granule_label.astype(np.int8)

    # This code is for the auxilary distance transform, move to dataloader

    # dist_labels = -1 * np.ones_like(data, dtype=np.float32)
    # pos_dist = ndimage.distance_transform_edt(label)
    # neg_dist = ndimage.distance_transform_edt(1 - label)
    # dist = np.where(label == 1, pos_dist, -1 * neg_dist)
    # dist_labels[idx] = dist / 720


def generate_data(sample, tomo_name, slices, z_limits):
    input_tomo_path = os.path.join(config.FLOAT_TOMO_DIR, sample, tomo_name)
    output_tomo_path = os.path.join(config.TRAIN_TOMO_DIR, sample, tomo_name)
    z_min, z_max = z_limits

    with h5py.File(input_tomo_path) as fh:
        data = fh["data"][()]

    mito_labels = -1 * np.ones_like(data, dtype=np.int8)
    granule_labels = -1 * np.ones_like(data, dtype=np.int8)

    mito_labels[:z_min] = 0
    mito_labels[z_max:] = 0
    granule_labels[:z_min] = 0
    granule_labels[z_max:] = 0

    for idx in slices:
        in_path = os.path.join(config.HASTY_ANNOTATIONS, sample, f"{tomo_name[:-4]}_{idx}.png")
        label = io.imread(in_path)
        insert_labels(label, idx, mito_labels, granule_labels)

    with h5py.File(output_tomo_path, 'w') as fh:
        fh.create_dataset("data", data=data)
        fh.create_dataset("mito", data=mito_labels, compression="gzip")
        fh.create_dataset("granule", data=granule_labels, compression="gzip")


def main():
    args = docopt(__doc__)
    sample = args["<sample>"]
    annotation_file = os.path.join(config.ANNOTATION_CSV, f"{sample}.csv")

    if not os.path.exists(annotation_file):
        raise RuntimeError(f"No annotations for {sample} were found")

    dst_dir = os.path.join(config.TRAIN_TOMO_DIR, sample)
    os.makedirs(dst_dir, exist_ok=True)
    df = pd.read_csv(annotation_file)
    
    for row in tqdm(df.itertuples()):
        generate_data(
            sample,
            tomo_name=row[1],
            slices=row[4:],
            z_limits=row[2:4]
        )

        
if __name__ == '__main__':
    main()
