"""Mitochondria Segmentation

Usage:
    mito_seg.py -i input.hdf -o output.hdf [-c] [-b] [-q]
    mito_seg.py (-h | --help)
    mito_seg.py --version

Options:
    -i <file>, --input <file>     Path to the input HDF file
    -o <file>, --output <file>    Path to the output file
    -c, --copy                    Copy the source data to the output file
    -b, --bin                     Bin the input by a factor of 2
    -q, --quiet                   Quite mode - suppress warnings and info
    -h, --help                    Show this screen.                    
"""
import sys
import logging
from rich.logging import RichHandler

from docopt import docopt
from pathlib import Path
from segmentation_utils import segment_mitochondria


def main():
    # parse args
    args = docopt(__doc__, version="Mitochondria Segmentation v0.1.0")
    input_path = Path(args["--input"])
    output_path = Path(args["--output"])
    copy = args["--copy"]
    bin2 = args["--bin"]
    level = "ERROR" if args["--quiet"] else "INFO"
    
    # setup logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )

    log = logging.getLogger("rich")
    log.info(f"Input source is {input_path}")
    log.info(f"Optional parameters --bin:{bin2}, --copy:{copy}")

    # validate input and output paths
    if not input_path.exists():
        log.error(f"Invalid input path: {input_path}")
        sys.exit(1)

    if output_path.exists():
        log.warning(f"Output file {output_path} exists and will be over-written")

    # run the mitochondria segmentation model
    segment_mitochondria(input_path, output_path, copy, bin2)
    

if __name__ == '__main__':
    main()
