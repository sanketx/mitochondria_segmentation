"""
Global config file for mito_methods project
"""

DATASETS = [
    "Q18",
    "Q20",
    "Q53",
    "Q66",
    "Q77",
    "Q109",
    "WT",
    "BACHD",
    "BACHD_controlRNA",
    "dN17_BACHD",
]

# Path to original high res Huntington tomograms
HD_DIR = "/sdf/home/s/sanketg/C036/20210811_mitochondria_for_joy/for_huntinton_paper"

# Project root directory
ROOT = "/sdf/home/s/sanketg/projects/mito_methods"

# Old project root directory
OLD_ROOT = "/sdf/home/s/sanketg/projects/mitochondria"

# Path to raw tomograms and mito / granule predictions
OLD_TOMO_DIR = f"{OLD_ROOT}/stats/new_merged"

# Path to preprocessed tomograms and granule predictions
RAW_TOMO_DIR = f"{ROOT}/data/tomograms/granule_data"

# Path to updated raw data - float 32 with proper distribution alignment
FLOAT_TOMO_DIR = f"{ROOT}/data/tomograms/float32_512" 

# Path to csv with annotation information
ANNOTATION_CSV = f"{ROOT}/data/csv"

# Path to extracted image slices
IMG_SLICES_DIR = f"{ROOT}/data/images/slices"

# Path to mitochondria and granule masks
HASTY_ANNOTATIONS = f"{ROOT}/data/images/hasty_masks"

# Path to granule masks
GRANULE_ANNOTATIONS = f"{ROOT}/data/images/granule_masks_autogen"

# Path to training data with new hasty.ai annotations
TRAIN_TOMO_DIR = f"{ROOT}/data/tomograms/hasty_train"

# Path to datasets splits
EXP_ROOT = f"{ROOT}/experiments"
