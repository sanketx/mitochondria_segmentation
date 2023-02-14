![a picture of mitochondria with colored granules](media/cover_1920x1080.jpg)

# 3D Segmentation of Mitochondrial Structures

A Deep Learning package for accurate segmentation of mitochondria and granules captured using cryo-electron tomography. This repository contains code for our papers
- **[CryoET Reveals Organelle Phenotypes in Huntington Disease Patient iPSC-Derived and Mouse Primary Neurons](https://www.nature.com/articles/s41467-023-36096-w)**
- **Robust and Label-Efficient Segmentation of Mitochondrial Structures in Cryo-electron Tomograms**

If you use these scripts or data for your research, please cite as
```bibtex
@article{wu2023cryoet,
  title={CryoET reveals organelle phenotypes in huntington disease patient iPSC-derived and mouse primary neurons},
  author={Wu, Gong-Her and Smith-Geater, Charlene and Galaz-Montoya, Jes{\'u}s G and Gu, Yingli and Gupte, Sanket R and Aviner, Ranen and Mitchell, Patrick G and Hsu, Joy and Miramontes, Ricardo and Wang, Keona Q and others},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={692},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## Features
1. MitoSeg is a toolkit for performing quantitative analyses of mitochondrial structures at scale. It currently supports segmentation of mitochoindria and granules.
2. MitoSeg is powered by a 3D-UNet which is trained on a diverse set of samples spanning multiple conditions and is capable of generalizing well to novel samples.
3. The trained models can be used out-of-the-box for segmentation using the included inference scripts. We also provide tools for fine-tuning models for new datasets.
4. Models are pre-trained using self-supervision and can be finetuned by labeling as few as 5 slices per tomogram.

## Getting Started

### Requirements
#### Software Requirements
- PyTorch v1.12.1
- CUDA Toolkit 11.6
- Additional packages contained in `environment.yml`
- Software dependencies are managed using Anaconda

#### Hardware Requirements
Our models run on NVIDIA GPUs and are quite memory intensive, so we would recommend using a GPU with a high memory capacity.

## Installation
1. Clone the repository to your computer. You can also download a zip file from GitHub.
```
git clone git@github.com:sanketx/mitochondria_segmentation.git
```
2. Create a new conda environment named `mito_seg` and activate it.
```
cd mitochondria_segmentation
conda env create -f environment.yml
conda activate mito_seg
```
3. Install the segmentation scripts in the `src` folder
```
conda develop src
```
4. Test the installation by printing the version number
```
python -m mito_seg --version
```

## Usage
The command `python -m mito_seg --help` will show you the usage instructions.
```
Mitochondria Segmentation

Usage:
    mito_seg.py -i input.hdf -o output.hdf [-c] [-b] [-q]
    mito_seg.py (-h | --help)
    mito_seg.py --version

Options:
    -i <file>, --input <file>     Path to the input HDF file
    -o <file>, --output <file>    Path to the output file
    -c, --copy                    Copy the source data to the output file
    -b, --bin                     Bin the input by a factor of 2
    -q, --quiet                   Quiet mode - suppress warnings and info
    -h, --help                    Show this screen.
```

The `mito_seg` program takes an input tomogram, runs it through the segmentation model, and writes the result to a specified output file. It expects inputs which are binned by 8 versions of the full resolution tomographic reconstruction. Since these are usually binned by 4, the `-b` flag can be used to tell `mito_seg` to further bin the input by a factor of 2. Additionally, the `-c` flag is used to copy the input data and store it alonside the segmentation results.

Note that the input tomograms are expected to have pixel intensities which follow a standard normal distribution. `mito_seg` clips them to +/-3 standard deviations. We provide an example tomogram, `BACHD_bin4.hdf`, to test the `mito_seg` program - it can be downloaded [here](https://drive.google.com/file/d/1BmPKXe9IxordM3NWKHoOC8gc9nGmm2oq/view?usp=share_link).

To run `mito_seg` on this tomogram, run the following command
```
python -m mito_seg --input BACHD_bin4.hdf -o result.hdf -bc
```
It will generate the following output and write the result to `result.hdf`
```
[08:51:51 PM] INFO     Input source is BACHD_bin4.hdf                                                                                           mito_seg.py:43
              INFO     Optional parameters --bin:True, --copy:True                                                                              mito_seg.py:44
              WARNING  Output file result.hdf exists and will be over-written                                                                   mito_seg.py:52
              INFO     Reading tomogram of shape (256, 1024, 1024) from /MDF/images/0/image                                           segmentation_utils.py:40
[08:51:52 PM] INFO     Tomogram binned by 2 from (256, 1024, 1024) -> (128, 512, 512)                                                 segmentation_utils.py:51
              INFO     Model loaded from checkpoint /home/sanket/Desktop/mitochondria_segmentation/models/mito_weights.pt             segmentation_utils.py:82
[08:51:53 PM] INFO     Predicting mitochondria probabilities in the tomogram                                                          segmentation_utils.py:95
[08:51:54 PM] INFO     Saving mitochondria predictions to result.hdf:/mito_pred                                                      segmentation_utils.py:116
              INFO     Copying original tomogram to result.hdf:/data                                                                 segmentation_utils.py:120

```
