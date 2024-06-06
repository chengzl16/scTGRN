# Inferring Gene Regulatory Networks from Single-Cell Time-Course Data Based on Temporal Convolutional Networks
# 1.scTGRN overview
![](https://github.com/chengzl16/scTGRN/blob/main/scTGRN.jpg)
# 2.The network structure of scTGRN(ATCN)
![](https://github.com/chengzl16/scTGRN/blob/main/ATCN.jpg)
# 3.scRNA-seq datasets
## 3.1 Evaluation of scTGRN on real datasets
Four real datasets：two of the real datasets are from mouse embryonic stem cells (mESC1 and mESC2) and two are from human embryonic stem cells (hESC1 and hESC2). Gene expression data of mESC1, mESC2, hESC1, hESC2 available at GSE79578,  GSE79578, E-MTAB-3929, GSE75748, respectively.
## 3.2 Evaluation of scTGRN on simulated datasets
Four simulation datasets: sim1, sim2, sim3, sim4.
## 3.3 Gene function assignment using scTGRN
Mouse brain cortex: gene expression data of mouse brain cortexare available at GSE104158.
## 3.4 Download link
Normalized expression data(real and simulation) can be downloaded from [link](https://doi.org/10.5281/zenodo.6720690 "title text") directly. (based on https://github.com/ericcombiolab/dynDeepDRIM)

# 4.Code Environment
The conda environment installation file "scTGRN_environment.yaml" can be downloaded from this repository.

# 5.scTGRN: infer the regulatory relationships of TF-gene pairs
## 5.1 Data preprocessing
* python script: 4D_input_tensors_construction.py<br>
According to the selected positive and negative gene pair samples and the normalized gene expression matrix, we construct the 4D input tensor of each gene for the sample.<br>
```
python 4D_input_tensors_construction.py
```
The input data of scTGRN include: datasetName_gene_list_ref.txt, datasetName_gene_pairs_400.txt, datasetName_gene_pairs_400_num.txt, gene expression matrix(need to download through the previous website).
* datasetName_gene_list_ref.txt: This file is a list of gene names from gene expression data converted to gene symbol ids. e.g., mesc1_gene_list_ref.txt. Format: "gene symbol ids\t sc gene id".
* datasetName_gene_pairs_400.txt: This file is a list containing TF-gene pairs and their labels. e.g., mesc1_gene_pairs_400.txt. Format: "geneA\t    GeneB\t   label".
* datasetName_gene_pairs_400_num.txt: This file is a list of indexes partitioned by the number of TF-gene pairs, where each number represents the index number corresponding to each TF. e.g., mesc1_gene_gene_pairs_400_num.txt. Format: "0\n  7136".
* gene expression matrix: This file is the gene expression matrix for all genes at different time points.

### The specific process of data preprocessing.
### Step1: Calculate a new "gene" called "AVG".
AVG: represents the average expressions of all genes in each cell.
### Step2: Constructor the genome.
gene "AVG" and gene pair (a, b) can be reconstructed into genome (a, b, AVG). 
### Step3: Divide the range of gene expression values equally.
We divide the range of expression values of the genome (a, b, AVG) in all cell samples into 8 equal bins, thus constructing an 8 × 8 × 8 3D matrix for the genome (a, b, AVG).
### Step4: Calculate the probability of co-occurs.
Each entry (i, j, k) in the 3D matrix represents the probability that (a, b, AVG) co-occurs at the i-th, j-th, and k-th expression levels.
### Step5: Construct the 4D input tensors(T × 8 × 8 × 8).
Perform the above operations for the gene expression matrix at each time point.

## 5.2 Training and test ATCN model.
* python script: training.py<br>
`training.py` uses the 4D input tensor to train and test the ATCN model. The training set and test set are randomly divided by 8:2. The file path needs to be changed before training and before testing.<br>
```
python training.py
```
File path of the 4D input tensor: datasetName_4D_input_tensors. e.g., mesc1_4D_input_tensors.<br>
