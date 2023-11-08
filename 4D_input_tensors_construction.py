import pandas as pd
from numpy import *
import os
from pandas import DataFrame

# Create folders for 4D input tensors
save_dir = os.getcwd() + '\\mesc1_4D_input_tensors'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
else:
    print(save_dir + "already exist.")

# Load the gene pair file
def get_gene_list(file_name):
    import re
    h = {}
    s = open("D:\\Python\\scTGRN\\database\\mesc1\\{}".format(file_name), 'r')
    for line in s:
        search_result = re.search(r'^([^\s]+)\s+([^\s]+)', line)
        h[search_result.group(1).lower()] = search_result.group(2)
    s.close()
    return h

# Load the gene pair file
def get_sepration_index (file_name):
    import numpy as np
    index_list = []
    s = open("D:\\Python\\scTGRN\\database\\mesc1\\{}".format(file_name), 'r')
    for line in s:
        index_list.append(int(line))
    return (np.array(index_list))

h_gene_list = get_gene_list('mesc1_gene_list_ref.txt')
new_gene = {'avg': 'avg'}
h_gene_list.update(new_gene)
print('read sc gene list')

# Load the gene expression data
sample_size_list = []
sample_sizex = []
total_RPKM_list = []
total_RPKM_gene_name = []
num_time_points = 9
for indexy in range(0, num_time_points):
    data = pd.read_csv("D:\\Python\\scTGRN\\database\\mesc1\\prep_three_RPKM" + '_' + str(indexy) + '.csv', encoding='gbk', sep=",")
    # print(data)
    data.set_index('Unnamed: 0', inplace=True)
    gene_name = list(data)
    total_RPKM_gene_name.extend(gene_name)
    k = data.shape[0]
    print(k)
    new_data = data.replace(0, NaN)
    row_mean = DataFrame.mean(new_data, axis=1, skipna=True)
    data['avg'] = row_mean
    total_RPKM_list.append(data)
    sample_size_list = sample_size_list + [indexy for i in range(data.shape[0])]
    sample_sizex.append(data.shape[0])
    samples = array(sample_size_list)
    sample_size = len(sample_size_list)
total_RPKM = pd.concat(total_RPKM_list, ignore_index=True)
total_RPKM = total_RPKM.replace(NaN, 0)
total_RPKM_gene_name = list(set(total_RPKM_gene_name))
total_RPKM_gene_name.append('avg')
print('read sc RNA-seq expression', sample_size)

# construction 4D input tensors
ground_truth_lable = 1
gene_pair_label = []
s = open("D:\\Python\\scTGRN\\database\\mesc1\\mesc1_gene_pairs_400.txt")
for line in s:
    line_index = line.find('\t')
    new_line = line[:line_index] + '\tavg' + line[line_index:]
    gene_pair_label.append(new_line)
gene_pair_index = get_sepration_index("mesc1_gene_pairs_400_num.txt")
s.close()
gene_pair_label_array = array(gene_pair_label)
for i in range(len(gene_pair_index)-1):
    # print(i)
    start_index = gene_pair_index[i]
    end_index = gene_pair_index[i+1]
    x = []
    y = []
    z = []
    for gene_pair in gene_pair_label_array[start_index:end_index]:
        separation = gene_pair.split()
        if ground_truth_lable == 1:
            x_gene_name, y_gene_name, z_gene_name, label = separation[0], separation[1], separation[2], separation[3]
            gene_pair_name = []
            gene_pair_name.append(x_gene_name)
            gene_pair_name.append(y_gene_name)
            gene_pair_name.append(z_gene_name)
            m = len(gene_pair_name)
            res = any(gene_pair_name == total_RPKM_gene_name[i:i + m] for i in range(len(total_RPKM_gene_name) - m + 1))
            if res:
                y.append(label)
            else:
                print(len(y))
                continue
        else:
            x_gene_name, y_gene_name, z_gene_name = separation[0], separation[1], separation[2]
        z.append(x_gene_name + '\t' + y_gene_name + '\t' + z_gene_name)

        if h_gene_list[x_gene_name] in total_RPKM_gene_name:
            x_tf = log10(total_RPKM[h_gene_list[x_gene_name]] + 10 ** -2)
        else:
            y.pop()
            z.pop()
            continue
        if h_gene_list[y_gene_name] in total_RPKM_gene_name:
            x_mid = log10(total_RPKM[h_gene_list[y_gene_name]] + 10 ** -2)
        else:
            y.pop()
            z.pop()
            continue
        if h_gene_list[z_gene_name] in total_RPKM_gene_name:
            x_gene = log10(total_RPKM[h_gene_list[z_gene_name]] + 10 ** -2)
        else:
            y.pop()
            z.pop()
            continue
        datax = concatenate((x_tf[:, newaxis], x_mid[:, newaxis], x_gene[:, newaxis], samples[:, newaxis]), axis=1)
        H, edges = histogramdd(datax, bins=(8, 8, 8, num_time_points))
        HT = (log10(H / sample_sizex + 10 ** -3) + 3) / 3
        H2 = transpose(HT, (3, 0, 1, 2))
        x.append(H2)
    if (len(x)>0):
        xx = array(x)[:, newaxis, :, :, :, :]
    else:
        xx = array(x)
    save(save_dir+'/NTxdata_tf' + str(i) + '.npy', xx)
    if ground_truth_lable == 1:
        save(save_dir+'/ydata_tf' + str(i) + '.npy', array(y))
    save(save_dir+'/zdata_tf' + str(i) + '.npy', array(z))
