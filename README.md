# MHCfovea

MHCfovea, a deep learning-based framework, provides predictions of MHC-I-peptide binding and connects MHC-I alleles with binding motifs.

## Overview

First, the predictor, an ensemble model based on convolutional neural networks, was trained on 150 observed alleles, and 42 important positions were highlighted from MHC-I sequence (182 a.a.) using ScoreCAM. Second, we made predictions on 150 observed alleles and 12,858 unobserved alleles with a single peptide dataset (number: 254,742), and extracted positive predictions (score > 0.9) to generate binding motifs. Then, after clustering the N-terminal and C-terminal sub-motifs, we build hyper-motifs and allele signatures based on 42 important positions to reveal the relation between binding motifs and MHC-I sequences.

<p align="center"><img src="figures/overview.jpg" alt="" width="700"></p>

## Application

MHCfovea takes MHC-I alleles (all alleles in the IPD-IMGT/HLA database (version 3.41.0) are available) and peptide sequences as inputs to predict the binding probability. For each queried allele, MHCfovea provides the cluster information and allele information of N- and C-terminal clusters respectively.

- cluster information
  - hyper-motif: the pattern of binding peptides in a specific cluster
  - allele signature: the pattern of MHC-I alleles in a specific cluster
- allele information
  - sub-motif: the binding sub-motif of the queried allele
  - highlighted allele signature: the consensus residues of the allele signature and the queried allele

If you find MHCfovea useful in your research please cite:


## Installation
1. Python3 is required
2. Download/Clone MHCfovea
```
git clone https://github.com/kohanlee1995/MHCfovea.git
cd MHCfovea
```
3. Install reqiured package
```
pip3 install -r requirements.txt
```

## Usage
```
usage: predictor [-h] [--alleles ALLELES] [--get_metrics] input output_dir

    MHCfovea, an MHCI-peptide binding predictor. In this prediction process, GPU is recommended.

    Having two modes:
    1. specific mode: each peptide has its corresponding MHC-I allele in the input file; column "mhc" or "allele" is required
    2. general mode: all peptides are predicted with all alleles in the "alleles" argument

    Input file:
    only .csv file is acceptable
    column "sequence" or "peptide" is required as peptide sequences
    column "mhc" or "allele" is optional as MHC-I alleles

    Output directory contains:
    1. prediction.csv: with new column "score" for specific mode or [allele] for general mode
    2. interpretation: a directory contains interpretation figures of each allele
    3. metrics.json: all and allele-specific metrics (AUC, AUC0.1, AP, PPV); column "bind" as benchmark is required


positional arguments:
  input              The input file
  output_dir         The output directory

optional arguments:
  -h, --help         show this help message and exit
  --alleles ALLELES  alleles for general mode
  --get_metrics      calculate the metrics between prediction and benchmark
```


## Example

```
python3 mhcfovea/predictor.py example/input.csv example/output
```

#### input file

| sequence | mhc |
|---|---|
| NAPWAVTSL | B*07:02 |
| VNLPINGNGKQ | B*07:02 |
| RVAEFHTEL | B*07:02 |
| AAAGPGAAL | B*07:02 |
| APAAIPAL | B*07:02 |
| APAAYPREVAL | B*07:02 |
| APAGGAGAL | B*07:02 |
| APAGSIISL | B*07:02 |
| APAPGAPLL | B*07:02 |
| APAPGAPLLPL | B*07:02 |
| APAPSRGSVQV | B*07:02 |
| APAPSRGSVQVAL | B*07:02 |
| APFGLKPRSV | B*07:02 |
| APFLRIAF | B*07:02 |
| APGDYGRQAL | B*07:02 |
| APGEPGSAF | B*07:02 |
| APGGGPGTL | B*07:02 |
| APGPGLLL | B*07:02 |
| APHDYGLIL | B*07:02 |
| APHPSSWETL | B*07:02 |

#### output file

| sequence | mhc | score |
|---|---|---|
| NAPWAVTSL | B*07:02 | 0.919 |
| VNLPINGNGKQ | B*07:02 | 0.968 |
| RVAEFHTEL | B*07:02 | 0.94 |
| AAAGPGAAL | B*07:02 | 0.959 |
| APAAIPAL | B*07:02 | 0.979 |
| APAAYPREVAL | B*07:02 | 0.997 |
| APAGGAGAL | B*07:02 | 0.998 |
| APAGSIISL | B*07:02 | 0.997 |
| APAPGAPLL | B*07:02 | 0.999 |
| APAPGAPLLPL | B*07:02 | 0.999 |
| APAPSRGSVQV | B*07:02 | 0.994 |
| APAPSRGSVQVAL | B*07:02 | 0.992 |
| APFGLKPRSV | B*07:02 | 0.961 |
| APFLRIAF | B*07:02 | 0.913 |
| APGDYGRQAL | B*07:02 | 0.995 |
| APGEPGSAF | B*07:02 | 0.986 |
| APGGGPGTL | B*07:02 | 0.983 |
| APGPGLLL | B*07:02 | 0.943 |
| APHDYGLIL | B*07:02 | 0.973 |
| APHPSSWETL | B*07:02 | 0.989 |

#### interpretation figure

<p align="center"><img src="example/output/interpretation/B0702.png" alt="" width="600"></p>
