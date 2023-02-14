<div style="width:80%">
	![a picture of mitochondria with colored granules](media/cover_1920x1080.jpg)
</div>
# 3D Segmentation of Mitochondrial Structures

A Deep Learning package for accurate segmentation of mitochondria and granules captured using cryo-electron tomography. This repository contains code for our papers
- **CryoET Reveals Organelle Phenotypes in Huntington Disease Patient iPSC-Derived and Mouse Primary Neurons**
- **Robust and Label-Efficient Segmentation of Mitochondrial Structures in Cryo-electron Tomograms**

If you use these scripts or data for your research, please cite as
```bibtex
@article {Wu2022.03.26.485912,
	author = {Wu, Gong-Her and Smith-Geater, Charlene and Galaz-Montoya, Jes{\'u}s G. and Gu, Yingli and Gupte, Sanket R. and Aviner, Ranen and Mitchell, Patrick G. and Hsu, Joy and Miramontes, Ricardo and Wang, Keona Q. and Geller, Nicolette R. and Danita, Cristina and Joubert, Lydia-Marie and Schmid, Michael F. and Yeung, Serena and Frydman, Judith and Mobley, William and Wu, Chengbiao and Thompson, Leslie M. and Chiu, Wah},
	title = {CryoET Reveals Organelle Phenotypes in Huntington Disease Patient iPSC-Derived and Mouse Primary Neurons},
	elocation-id = {2022.03.26.485912},
	year = {2022},
	doi = {10.1101/2022.03.26.485912},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Huntington{\textquoteright}s Disease (HD) is caused by an expanded CAG repeat in the huntingtin gene, yielding a Huntingtin protein with an expanded polyglutamine tract. Patient-derived induced pluripotent stem cells (iPSCs) can help understand disease; however, defining pathological biomarkers is challenging. Here, we used cryogenic electron tomography to visualize neurites in HD patient iPSC-derived neurons with varying CAG repeats, and primary cortical neurons from BACHD, deltaN17-BACHD, and wild-type mice. In HD models, we discovered mitochondria with enlarged granules and distorted cristae, and thin sheet aggregates in double membrane-bound organelles. We used artificial intelligence to quantify mitochondrial granules, and proteomics to show differential protein content in HD mitochondria. Knockdown of Protein Inhibitor of Activated STAT1 ameliorated aberrant phenotypes in iPSC-neurons and reduced phenotypes in BACHD neurons. We show that integrated ultrastructural and proteomic approaches may uncover early HD phenotypes to accelerate diagnostics and the development of targeted therapeutics for HD.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2022/03/27/2022.03.26.485912},
	eprint = {https://www.biorxiv.org/content/early/2022/03/27/2022.03.26.485912.full.pdf},
	journal = {bioRxiv}
}

```

## Features
1. MitoSeg is a toolkit for performing quantitative analyses of mitochondrial structures at scale. It currently supports segmentation of mitochoindria and granules.
2. MitoSeg is powered by a 3D-UNet which is trained on a diverse set of samples spanning multiple conditions and is capable of generalizing well to novel samples.
3. The trained models can be used out-of-the-box for segmentation using the included inference scripts. We also provide tools for fine-tuning models for new datasets.
4. Models are pre-trained using self-supervision and can be finetuned by labeling as few as 5 slices per tomogram.

## Getting Started

### Requirements

- PyTorch
- Pytorch Lightning
- CUDA Toolkit 11.3
- Additional packages contained in `environment.yml`

## Usage
This repository is under active development - detailed instructions for usage will be uploaded soon.

