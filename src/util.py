import os, sys, re, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings('ignore')


"""""""""""""""""""""""""""""""""""""""""
# Encoding and Build Dataset
"""""""""""""""""""""""""""""""""""""""""
def BLOSUM62Encoder(seq: str, length: int, padding: bool) -> np.array:
    dict_map = {
        "A" : [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4],
        "R" : [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4],
        "N" : [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4],
        "D" : [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4],
        "C" : [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],
        "Q" : [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4],
        "E" : [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4],
        "G" : [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4],
        "H" : [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4],
        "I" : [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4],
        "L" : [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4],
        "K" : [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4],
        "M" : [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4],
        "F" : [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4],
        "P" : [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4],
        "S" : [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4],
        "T" : [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4],
        "W" : [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4],
        "Y" : [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4],
        "V" : [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4],
        "B" : [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4],
        "Z" : [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4],
        "X" : [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4],
        "." : [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1],
        "U" : [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1]
    }
    if padding == False:
        length = len(seq)

    arr = np.tile(np.array(dict_map["."]), (length, 1)).T
    
    for idx in range(len(seq)):
        arr[:, idx] = dict_map[seq[idx]]
    return arr


def OneHotEncoder(seq: str, length: int, padding: bool) -> np.array:
    dict_map = {
        'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    if padding == False:
        length = len(seq)

    arr = np.zeros((21, length))
    for idx in range(length):
        try:
            arr[:, idx] = dict_map[seq[idx]]
        except:
            arr[:, idx] = dict_map["."]
    return arr


def BuildDataset(df, encoding_method, epitope_length, with_label=True):
    x_list, y_list = list(), list()
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
        # epitope encoding
        if encoding_method == "onehot":
            epitope_encode = OneHotEncoder(row.sequence, epitope_length, True)
        elif encoding_method == "blosum":
            epitope_encode = BLOSUM62Encoder(row.sequence, epitope_length, True)
        else:
            print("wrong epitope encoding method")
            return None
        
        # x = epitpoe_encode
        x_list.append(epitope_encode)
        
        # y = [idx, classification_value, regression_value]
        if with_label:
            try:
                y_list.append([idx, row.bind, row.value])
            except:
                y_list.append([idx, row.bind])
        else:
            y_list.append([idx])

    x_tensor = torch.FloatTensor(x_list)
    y_tensor = torch.FloatTensor(y_list)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    
    return dataset


"""""""""""""""""""""""""""""""""""""""""
# Data for Training
"""""""""""""""""""""""""""""""""""""""""
class Data():
    def __init__(self, df_file, dataset_file, batch_size, shuffle, decoy_times=None):
        
        self.df_file = df_file
        self.dataset_file = dataset_file

        # list for [hit, decoy]
        if type(self.df_file) == list:
            df_list = list()
            for f in self.df_file:
                df_list.append(pd.read_csv(f, index_col=0))
            self.df = pd.concat(df_list)
        else:
            self.df = pd.read_csv(self.df_file, index_col=0)

        if type(self.dataset_file) == list:
            dataset_list = list()
            for f in self.dataset_file:
                dataset_list.append(torch.load(f))
            self.dataset = torch.utils.data.ConcatDataset(dataset_list)
        else:
            self.dataset = torch.load(self.dataset_file)

        # decoy_times, for validation
        if decoy_times != None:
            num = (self.df['source']=='MS').sum() * decoy_times
            decoy_idx = self.df[(self.df['source']=='protein_decoy') | (self.df['source']=='random_decoy')].index.to_numpy()
            np.random.seed(0)
            decoy_idx = np.random.choice(decoy_idx, num, replace=False)
            hit_idx = self.df[(self.df['source']=='MS') | (self.df['source']=='assay')].index.to_numpy()
            self.df = self.df.loc[np.concatenate((hit_idx, decoy_idx))]
            self.dataset = torch.utils.data.Subset(self.dataset, np.concatenate((hit_idx, decoy_idx)))
        
        self.mhc_idx = self.df['mhc']
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)


"""""""""""""""""""""""""""""""""""""""""
# Metrics
"""""""""""""""""""""""""""""""""""""""""
# AUC, AUC0.1, AP, PPV
def CalculateMetrics(y, pred):
    if len(set(y)) == 1:
        return {"AUC": np.nan, "AUC0.1": np.nan, "AP": np.nan, "PPV": np.nan}
    # AUC
    fpr, tpr, _ = metrics.roc_curve(y, pred)
    auc_score = np.around(metrics.auc(fpr, tpr), decimals=3)
    # AUC0.1
    idx = np.where(fpr <= 0.1)[0]
    auc01_score = np.around(np.trapz(tpr[idx], fpr[idx]) * 10, decimals=3)
    # AP
    avg_precision_score = np.around(metrics.average_precision_score(y, pred), decimals=3)
    # PPV
    num = sum(y==1)
    idx = pred.argsort()[::-1][:num]
    ppv = np.around(y[idx].sum() / num, decimals=3)

    return {"AUC": auc_score, "AUC0.1": auc01_score, "AP": avg_precision_score, "PPV": ppv}


def CalculateAlleleMetrics(mhc_list, y, pred):
    df = pd.DataFrame()
    df["mhc"] = mhc_list
    df["y"] = y
    df["pred"] = pred

    result_dict = dict()
    allele_list = df["mhc"].unique()
    for allele in allele_list:
        temp_df = df[df["mhc"]==allele]
        temp_y = temp_df["y"].to_numpy()
        temp_pred = temp_df["pred"].to_numpy()
        result_dict[allele] = CalculateMetrics(temp_y, temp_pred)

    return result_dict


"""""""""""""""""""""""""""""""""""""""""
# Others
"""""""""""""""""""""""""""""""""""""""""
def ConvertAllele(allele):
    if allele.startswith('HLA-'):
        allele = allele[4:]
    if re.match(r'[ABC][0-9]+', allele):
        allele = '%s*%s:%s'%(allele[0], allele[1:3], allele[3:])
    elif re.match(r'[ABC][0-9]+\:[0-9]+', allele):
        allele = '%s*%s:%s'%(allele[0], allele[1:3], allele[4:])
    # check
    if re.match(r'[ABC]\*[0-9]+\:[0-9]+', allele):
        return allele
    else:
        print('Allele name format error')
        assert ValueError