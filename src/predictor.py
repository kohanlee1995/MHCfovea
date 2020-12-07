import os, sys, re, json, importlib, random, copy, argparse
import numpy as np
import pandas as pd
from collections import OrderedDict 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logomaker as lm
from util import *
import warnings
warnings.filterwarnings('ignore')


class Predictor():
    def __init__(self, mhc_encode_file, model_file, model_state_files, encoding_method):
        # MHC binding domain encoding
        self.mhc_encode_dict = np.load(mhc_encode_file, allow_pickle=True)[()]

        # device: gpu or cpu
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.batch_size = 4096
        else:
            self.device = torch.device('cpu')
            self.batch_size = 64

        # model
        if encoding_method == 'onehot':
            dim = 21
        elif encoding_method == 'blosum':
            dim = 24
        else:
            print("Wrong encoding method")
            raise ValueError
        model_file = '.'.join(model_file.split('.')[0].split('/'))
        module = importlib.import_module(model_file)
        self.model = module.CombineModel(module.MHCModel(dim), module.EpitopeModel(dim))

        # model states
        self.models = OrderedDict()
        for i in range(len(model_state_files)):
            basename = re.split(r'[\/\.]', model_state_files[i])[-2]
            model_state_dict = torch.load(model_state_files[i], map_location=self.device)
            self.models[basename] = copy.deepcopy(self.model)
            self.models[basename].load_state_dict(model_state_dict['model_state_dict'])
            self.models[basename].to(self.device)


    def __call__(self, df, dataset, allele=None):
        result_df = pd.DataFrame(index=df.index, columns=list(self.models.keys()))
        result_df['sequence'] = df['sequence']

        # general mode
        if allele:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            preds = self._predict(allele, dataloader)
            result_df.loc[:, list(self.models.keys())] = preds

        # specific mode
        else:
            result_df['mhc'] = df['mhc']
            for allele, sub_df in tqdm(df.groupby('mhc'), desc='alleles', leave=False, position=0):
                idx = sub_df.index
                sub_dataset = torch.utils.data.Subset(dataset, idx)
                sub_dataloader = torch.utils.data.DataLoader(sub_dataset, batch_size=self.batch_size, shuffle=False)
                preds = self._predict(allele, sub_dataloader)
                result_df.loc[idx, list(self.models.keys())] = preds

        return result_df


    def _predict(self, allele, dataloader):
        mhc_encode = self.mhc_encode_dict[allele]
        df = pd.DataFrame()
        for key, model in tqdm(self.models.items(), desc='models', leave=False, position=1):
            for j, (x,y) in enumerate(tqdm(dataloader, desc='batches', leave=False, position=2)):
                with torch.no_grad():
                    model.eval()
                    num = x.shape[0]
                    epitope_encode = x.to(self.device).float()
                    mhc_encode_tile = torch.FloatTensor(np.tile(mhc_encode, (num, 1, 1))).to(self.device)
                    pred = model(mhc_encode_tile, epitope_encode).to('cpu')
                    pred = pred.view(-1,).numpy()
                    if j==0:
                        preds = pred
                    else:
                        preds = np.append(preds, pred, axis=0)
            df[key] = preds
        return df.values


def GetSeqlogoDF(seqs: pd.Series, sub_motif_len=4):
    aa_str = 'ACDEFGHIKLMNPQRSTVWY'
    seqs = seqs.apply(lambda x: x[:sub_motif_len] + x[-sub_motif_len:])
    df = pd.DataFrame(columns=list(aa_str))
    seqlogo_df = lm.alignment_to_matrix(sequences=seqs, to_type='information', characters_to_ignore='XU.')
    df = pd.concat([df, seqlogo_df], axis=0)
    df = df[list(aa_str)]
    df = df.fillna(0.0)
    return df


def ArgumentParser():
    description = '''
    MHCI-peptide binding prediction
    Having two modes:
    1. specific mode: peptide file must contain "mhc" column
    2. general mode: use the "alleles" argument for all peptides
    Output directory contains:
    1. prediction.csv: with new column "score" for specific mode or [allele] for general mode
    2. motif.npy: dictionary with allele as key and motif array as value (number of positive samples >= 10)
    3. metrics.json: all and allele-specific metrics (AUC, AUC0.1, AP, PPV)
    4. tmp_prediction.csv: prediction of all models
    5. record.json: recording arguments
    '''
    parser = argparse.ArgumentParser(prog='predictor', description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    data_args = parser.add_argument_group('Data Arguments')
    data_args.add_argument('--mhc_encode_file', required=True, help='MHCI sequence encoding file')
    data_args.add_argument('--peptide_dataframe', required=True, help='csv file, contains "sequence" and "mhc" columns')
    data_args.add_argument('--peptide_dataset', required=False, default=None, help='dataset file built from the "BuildDataset" function in "util.py", default=None')
    data_args.add_argument('--encoding_method', required=False, type=str, default='onehot', help='onehot or blosum, default=onehot')
    data_args.add_argument('--alleles', required=False, default=None, help='alleles for all peptides, default=None')

    model_args = parser.add_argument_group('Model Arguments')
    model_args.add_argument('--model_file', required=True, help='model architecture file from the same directory')
    model_args.add_argument('--model_state_dir', required=True, help='model state directory')
    model_args.add_argument('--model_num', required=False, type=int, default=100, help='the number of model state used for prediction, default is all avaliable models')
    
    other_args = parser.add_argument_group('Other Arguments')
    other_args.add_argument('--output_dir', required=True, help='output directory')
    other_args.add_argument('--seqlogo_threshold', required=False, type=float, default=0.9, help='prediction threshold for building seqlogo dataframe, default=0.9')
    other_args.add_argument('--get_metrics', required=False, default=False, action='store_true', help='calculate the metrics, and peptide data must have labels')
    other_args.add_argument('--save_tmp', required=False, default=False, action='store_true', help='save temporary file')

    return parser


if __name__=="__main__":
    """""""""""""""""""""""""""""""""""""""""
    # Arguments
    """""""""""""""""""""""""""""""""""""""""
    args = ArgumentParser().parse_args()

    # data
    mhc_encode_file = args.mhc_encode_file
    peptide_dataframe = args.peptide_dataframe
    peptide_dataset = args.peptide_dataset
    encoding_method = args.encoding_method
    alleles = args.alleles
    
    # model
    model_file = args.model_file
    model_state_dir = args.model_state_dir
    model_num = args.model_num
    
    # others
    output_dir = args.output_dir
    seqlogo_threshold = args.seqlogo_threshold
    positive_threshold = 10
    get_metrics = args.get_metrics
    save_tmp = args.save_tmp

    # mkdir output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    """""""""""""""""""""""""""""""""""""""""
    # Loading Data & Model
    """""""""""""""""""""""""""""""""""""""""
    print("Loading data and model...")

    # model state files
    model_state_files = list()
    for file in os.listdir(model_state_dir):
        model_state_files.append('%s/%s'%(model_state_dir, file))
    model_state_files.sort()
    if model_num > len(model_state_files):
        model_num = len(model_state_files)
        print("model_num argument is more than the number of available models, so use all available models")
    model_state_files = model_state_files[:model_num]

    # peptide dataframe
    df = pd.read_csv(peptide_dataframe)
    cols = ['sequence']
    if not alleles:
        df['mhc'] = df['mhc'].apply(lambda x: ConvertAllele(x))
        cols.append('mhc')
    if get_metrics:
        cols.append('bind')
    df = df[cols]
    
    if peptide_dataset:
        dataset = torch.load(peptide_dataset)
    else:
        dataset = BuildDataset(df, 'onehot', 15, with_label=get_metrics)

    
    """""""""""""""""""""""""""""""""""""""""
    # Prediction
    """""""""""""""""""""""""""""""""""""""""
    print("Predicting...")

    # predictor
    Pred = Predictor(mhc_encode_file, model_file, model_state_files, encoding_method)

    # seqlogo dict
    seqlogo_dict = dict()

    # general mode
    if alleles:
        metrics_dict = dict()
        for allele in tqdm(args.alleles.split(','), desc='alleles', leave=False, position=0):
            allele = ConvertAllele(allele)
            pred_df = Pred(df, dataset, allele=allele)
            df[allele] = pred_df[list(Pred.models.keys())].mean(axis=1)
            if save_tmp:
                if get_metrics:
                    pred_df['bind'] = df['bind']
                pred_df.to_csv('%s/tmp_prediction_%s.csv'%(output_dir, allele[0]+allele[2:4]+allele[5:]))

            # seqlogo
            idx = np.where(df[allele] > seqlogo_threshold)[0]
            if len(idx) >= positive_threshold:
                seqlogo_dict[allele] = GetSeqlogoDF(df.iloc[idx]['sequence']).values

            # metrics
            if get_metrics:
                metrics_dict[allele] = CalculateMetrics(df['bind'], df[allele])

    # specific mode
    else:
        pred_df = Pred(df, dataset)
        df['score'] = pred_df[list(Pred.models.keys())].mean(axis=1)
        if save_tmp:
            if get_metrics:
                pred_df['bind'] = df['bind']
            pred_df.to_csv('%s/tmp_prediction.csv'%output_dir)
        
        # seqlogo
        for allele, sub_df in df.groupby('mhc'):
            idx = np.where(sub_df['score'] > seqlogo_threshold)[0]
            if len(idx) >= positive_threshold:
                seqlogo_dict[allele] = GetSeqlogoDF(sub_df.iloc[idx]['sequence']).values

        # metrics
        if get_metrics:
            all_metrics = CalculateMetrics(df['bind'], df['score'])
            allele_metrics = CalculateAlleleMetrics(df['mhc'], df['bind'], df['score'])
            metrics_dict = {'all': all_metrics, **allele_metrics}


    """""""""""""""""""""""""""""""""""""""""
    # Save result and record
    """""""""""""""""""""""""""""""""""""""""
    # result
    df.to_csv('%s/prediction.csv'%output_dir)
    np.save('%s/motif.npy'%output_dir, seqlogo_dict)
    if get_metrics:
        json.dump(metrics_dict, open('%s/metrics.json'%output_dir, 'w'))

    # record
    record_dict = dict({
        'mhc_encode_file': mhc_encode_file,
        'peptide_dataframe': peptide_dataframe,
        'peptide_dataset': peptide_dataset,
        'encoding_method': encoding_method,
        'alleles': alleles,
        'model_state_dir': model_state_dir,
        'model_num': model_num,
        'seqlogo_threshold': seqlogo_threshold
        })

    json.dump(record_dict, open('%s/record.json'%output_dir, 'w'))