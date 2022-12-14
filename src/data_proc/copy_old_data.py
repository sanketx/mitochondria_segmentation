"""Copy Old Data.

Mostly DEPRECATED, don't run

This program copies old data from OLD_DATA_DIR to RAW
Raw tomograms are copied along with granule predictions
Some are preprocessed to normalize contrast

Specify sample as "All" to copy all samples
Otherwise specify individual samples

Usage:
  copy_old_data.py <sample>
  copy_old_data.py (-h | --help)

Options:
  -h --help         Show this screen

"""
import os
import h5py
import numpy as np
from docopt import docopt

from .. import config

preproc = {
    "BACHD": True,
    "BACHD_controlRNA": False,
    "BACHD_pias1": False,
    "dN17_BACHD": True,
    "Q18": True,
    "Q53": True,
    "Q66": False,
    "Q66_KD": False,
    "Q77": False,
    "Q109": False,
    "WT": True,
    "Q20": False,
}


def normalize_contrast(data):
    mean = data.mean()
    delta = 3 * data.std()
    data = np.clip(data, mean - delta, mean + delta)
    data = 255 * (data - data.min()) / np.ptp(data)
    return data.astype(np.uint8)


def copy_file(src_path, dst_path, preprocess):
    with h5py.File(src_path) as fh:
        data = fh["data"][()]
        granule = fh["granule"][()]

    if preprocess:
        data = normalize_contrast(data)

    with h5py.File(dst_path, 'w') as fh:
        fh.create_dataset("data", data=data, compression="gzip")
        fh.create_dataset("granule", data=granule, compression="gzip")


def copy_samples(sample):
    src_dir = os.path.join(config.OLD_TOMO_DIR, sample)
    dst_dir = os.path.join(config.RAW_TOMO_DIR, sample)
    os.makedirs(dst_dir, exist_ok=True)

    for file_name in os.listdir(src_dir):
        print(f"Copying {file_name}")

        copy_file(
            os.path.join(src_dir, file_name),
            os.path.join(dst_dir, file_name),
            preproc[sample]
        )


def main():
    args = docopt(__doc__)
    sample = args["<sample>"]

    if sample in preproc:
        copy_samples(sample)

    elif sample == "All":
        for sample in preproc:
            copy_samples(sample)

    else:
        print(f"Invalid sample: {sample}")


if __name__ == '__main__':
    main()
