# MHCfovea

MHCfovea integrates a supervised prediction module and an unsupervised summarization module to connect important residues to binding motifs.

## Overview

First, the predictor, an ensemble model of multiple convolutional neural networks (CNN model), was trained on 150 observed alleles. In the predictor, 42 important positions were highlighted from MHC-I sequence (182 a.a.) using ScoreCAM. Next, we made predictions on 150 observed alleles and 12,858 unobserved alleles against a peptide dataset (number: 254,742), and extracted positive predictions (score > 0.9) to generate the binding motif of an allele. Then, after clustering the N-terminal and C-terminal sub-motifs, we build hyper-motifs and the corresponding allele signatures based on 42 important positions to reveal the relation between binding motifs and MHC-I sequences. The resultant pairs of hyper-motifs and allele signatures can be easily queried through a web interface (https://mhcfovea.ailabs.tw)

<p align="center"><img src="figures/overview.png" alt="" width="800"></p>

## Application

MHCfovea takes MHC-I alleles (all alleles in the IPD-IMGT/HLA database (version 3.41.0) are available) and peptide sequences as inputs to predict the binding probability. For each queried allele, MHCfovea provides the cluster information and allele information of N- and C-terminal clusters respectively.

- cluster information
  - hyper-motif: the pattern of binding peptides in a specific cluster
  - allele signature: the pattern of MHC-I alleles in a specific cluster
- allele information
  - sub-motif: the binding sub-motif of the queried allele
  - highlighted allele signature: the consensus residues of the allele signature and the queried allele

If you find MHCfovea useful in your research please cite:
<div>
  <i>Lee, K.-H., Chang, Y.-C., Chen, T.-F., Juan, H.-F., Tsai, H.-K., Chen, C.-Y.<sup>*</sup></i>
  <b>Connecting MHC-I-binding motifs with HLA alleles via deep learning.</b> Manuscript submitted for publication
</div>


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
| PVPTYGLSV | B*07:02 |
| APGARNTAAVL | B*07:02 |
| SPAPPTCHEL | B*07:02 |
| PGLAVKELK | B*07:02 |
| GPMVAGGLL | B*07:02 |

#### output file

| sequence | mhc | score |
|---|---|---|
| PVPTYGLSV | B*07:02 | 0.606 |
| APGARNTAAVL | B*07:02 | 0.987 |
| SPAPPTCHEL | B*07:02 | 0.997 |
| PGLAVKELK | B*07:02 | 0.569 |
| GPMVAGGLL | B*07:02 | 0.966 |

#### interpretation figure

<p align="center"><img src="example/output/interpretation/B0702.png" alt="" width="800"></p>
