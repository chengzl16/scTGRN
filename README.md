# Inferring Gene Regulatory Networks from Single-Cell Time-Course Data Based on Temporal Convolutional Networks

# 1.scRNA-seq datasets
## 1.1 Evaluation of scTGRN on real datasets
Four real datasets：two of the real datasets are from mouse embryonic stem cells (mESC1 and mESC2) and two are from human embryonic stem cells (hESC1 and hESC2). Gene expression data of mESC1, mESC2, hESC1, hESC2 available at GSE79578,  GSE79578, E-MTAB-3929, GSE75748, respectively.
## 1.2 Evaluation of scTGRN on simulated datasets
Four simulation datasets: sim1, sim2, sim3, sim4.
## 1.3 Gene function assignment using scTGRN
Mouse brain cortex: gene expression data of mouse brain cortexare available at GSE104158.
## 1.4 Download link
Normalized expression data(real and simulation) can be downloaded from [link](https://doi.org/10.5281/zenodo.6720690 "title text") directly. (based on https://github.com/ericcombiolab/dynDeepDRIM)

# 2.Code Environment
The conda environment installation file "scTGRN_environment.yaml" can be downloaded from this repository.

# 3.scTGRN: infer the regulatory relationships of TF-gene pairs
The input data of scTGRN include: datasetName_gene_list_ref.txt, datasetName_gene_pairs_400.txt, datasetName_gene_pairs_400_num.txt, gene expression matrix(need to download through the previous website).
[--]
## 3.1 Data preprocessing



