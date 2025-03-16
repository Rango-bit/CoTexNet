<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<p align="center">
  <h2 align="center" style="margin-top: -30px;">Learning conceptual text prompts from visual regions of interest for medical image segmentation</h2>
</p>

![Version](https://img.shields.io/badge/Version-1.0-lightgrey.svg) 
![LastUpdated](https://img.shields.io/badge/LastUpdated-2025.03-lightblue.svg)

## 📋 Overview

This repository contains a official implementation implementation of our research paper “Learning conceptual textual prompts from visual regions of interest for medical image segmentation”. Our approach utilizes learning conceptual text from ROI images to assist the VLSM model for visual segmentation.

## Framework

<p align="center">
  <img src="images/Figure.jpg" alt="Logo" style="width:80%;">
</p>

## 🛠️ Installation

### Setup Environment
```bash
# Create and activate conda environment
conda create -n vlsm python=3.10
conda activate vlsm

# Install dependencies
pip install -r requirements.txt
```

<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

### Datasets Links
| Dataset  | Download | Dataset  | Download 
|-------|-------|-------|-------|
| Kvasir | [Download](https://datasets.simula.no/kvasir-seg/) |DFU|[Download](https://dfu-challenge.github.io/dfuc2021.html) |
| BKAI | [Download](https://www.kaggle.com/c/bkai-igh-neopolyp/data) |ISIC|[Download](https://challenge.isic-archive.com/data/#2018) |
| CLinicDB | [Download](https://polyp.grand-challenge.org/CVCClinicDB/) |GLaS|[Download](https://paperswithcode.com/dataset/glas) |
| BUSI | [Download](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) |CAMUS|[Download](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html) |
| DDTI | [Download](http://cimalab.unal.edu.co/?lang=en&mod=project&id=31)  |ACDC|[Download](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) |

<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

### Dataset Preparation

Please prepare the dataset in the following format to facilitate the use of the code:

```angular2html
├── data_process
   ├── kvasir
   │   ├── images
   |   |   
   │   ├── masks
   |   | 
   │   └── ROIs
   |  
   └── clinicdb
       ├── images
       ......
```

Using mask labels to cover the original images to get the ROI images:
```bash
python data_process/make_ROI_img.py \
    --dataset-name kvasir \
    --model-path kvasir \
    --img-format .jpg \ # Image Storage Format
```

<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

### Models
- LLaVA-1.5 (`llava`: used to get textual attribute descriptions from ROIs)
- CLIPSeg(`clipseg`: model backbone)

