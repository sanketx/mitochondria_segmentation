"""Generate Annotation CSV

Generates skeleton CSVs in which annotation information is entered
Each row contains a tomo name and a list of slices to be annotated
and the min and max Z limits - no mitochondria are visible beyond them
By default, we label 5 slices per tomogram

Usage:
  generate_annotation_csv.py <sample> [--num_slices=<n>]
  generate_annotation_csv.py (-h | --help)

Options:
  -h --help         Show this screen
  --num_slices=<n>  Number of slices [default: 5]

"""
import os
import pandas as pd
from docopt import docopt

from .. import config

from collections import defaultdict


def get_slices_from_images(sample):
    # Was used to recover lost data, no longer needed

    img_dir = os.path.join(config.IMG_SLICES_DIR, sample)
    image_list = os.listdir(img_dir)
    slices = defaultdict(list)

    for img_name in image_list:
        split_idx = img_name.rfind('_')
        tomo_name = f"{img_name[:split_idx]}.hdf"
        slice_idx = img_name[split_idx + 1:].split('.')[0]
        slices[tomo_name].append(int(slice_idx))

    for v in slices.values():
        v.sort()

    df = pd.DataFrame(data=slices).T.reset_index()
    return df.rename(columns={
        "index": "tomo_name",
        0: "slice_0",
        1: "slice_1",
        2: "slice_2",
        3: "slice_3",
        4: "slice_4",
    })


def main():
    args = docopt(__doc__)
    sample = args["<sample>"]
    num_slices = int(args["--num_slices"])
    columns = ["tomo_name", "z_min", "z_max"]

    for i in range(num_slices):
        columns.append(f"slice_{i}")

    src_dir = os.path.join(config.RAW_TOMO_DIR, sample)
    dst_path = os.path.join(config.ANNOTATION_CSV, f"{sample}.csv")
        
    if not os.path.exists(src_dir):
        raise RuntimeError(f"No tomograms were found for {sample}")

    if os.path.exists(dst_path):
        raise RuntimeError(f"An annotation CSV for {sample} already exists")

    file_list = os.listdir(src_dir)
    df = pd.DataFrame(data={"tomo_name": file_list})
    
    # recover lost data :(
    # df = get_slices_from_images(sample) 
    
    df = df.reindex(columns=columns).sort_values(by="tomo_name")
    df.to_csv(dst_path, index=False) 


if __name__ == '__main__':
    main()
