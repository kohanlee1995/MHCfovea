import os, sys, re, json, importlib, argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from cam import *
import warnings
warnings.filterwarnings('ignore')


class Masks():
    def __init__(self, df, dataset, mhc_dict, cam):
        self.df = df
        self.dataset = dataset
        self.mhc_dict = mhc_dict
        self.cam = cam
        self.mhc_masks = None
        self.epitope_masks = None
        self.idxs_cluster = None
        
    
    def __call__(self, allele, column, threshold, num, batch_size, logdir):
        self.logdir = logdir
        if not os.path.isdir(self.logdir):
            os.mkdir(self.logdir)
        
        temp_df = self.df[(self.df["mhc"]==allele) & (self.df[column]>threshold)]
        max_num = temp_df.shape[0]
        if num >= max_num:
            num = max_num
            self.idxs = temp_df.index
        else:
            self.idxs = np.random.choice(temp_df.index, num, replace=False)
        self.idxs = [self.df.index.get_loc(i) for i in self.idxs]
        
        batch_num = int(np.ceil(num / batch_size))
        for i in range(batch_num):
            mhc_masks, epitope_masks = self._get_masks(self.idxs[i*batch_size: (i+1)*batch_size])
            if i==0:
                self.mhc_masks = mhc_masks
                self.epitope_masks = epitope_masks
            else:
                self.mhc_masks = np.append(self.mhc_masks, mhc_masks, axis=0)
                self.epitope_masks = np.append(self.epitope_masks, epitope_masks, axis=0)
        
        if num != 0:    
            self._plot_heatmap()
            self._plot_clustermap()
            self._record()
    
    
    def _get_masks(self, idxs):
        mhc = torch.FloatTensor([self.mhc_dict[self.df.iloc[int(self.dataset[idx][1][0])]['mhc']] for idx in idxs])
        epitope = torch.FloatTensor([np.array(self.dataset[idx][0]) for idx in idxs]) 
        
        mhc_masks, epitope_masks = self.cam(mhc, epitope)
        
        # expand epitope masks
        if epitope_masks.shape[-1] != 0:
            epitope_arr = np.zeros((len(idxs), 10))
            for i in range(len(idxs)):
                length = int(self.df.iloc[idxs[i]]["peptide_length"])
                epitope_arr[i, :5] = epitope_masks[i, :5]
                epitope_arr[i, -5:] = epitope_masks[i, length-5:length]
        else:
            epitope_arr = epitope_masks
        
        return mhc_masks, epitope_arr
    
    
    def _plot_heatmap(self):
        if self.mhc_masks.shape[-1] != 0:
            plt.figure(figsize=(8,8))
            sns.heatmap(self.mhc_masks, cmap='Blues')
            plt.savefig("%s/mhc_heatmap.png"%self.logdir)
        
        if self.epitope_masks.shape[-1] != 0:
            plt.figure(figsize=(8,8))
            sns.heatmap(self.epitope_masks, cmap='Blues')
            plt.savefig("%s/epitope_heatmap.png"%self.logdir)
        
    
    def _plot_clustermap(self):
        if self.epitope_masks.shape[-1] != 0:
            g = sns.clustermap(self.epitope_masks, col_cluster=False, cmap='Blues')
            self.idxs_cluster = g.dendrogram_row.reordered_ind
            plt.savefig("%s/epitope_clustermap.png"%self.logdir)
        
    
    def _record(self):
        record_dict = {
            "idxs": np.array(self.idxs),
            "mhc_masks": self.mhc_masks,
            "epitope_masks": self.epitope_masks,
            "idxs_cluster": np.array(self.idxs_cluster)
        }
        np.save("%s/record.npy"%self.logdir, record_dict)


def ArgumentParser():
    description = '''
    Run GradCAM / GradCAM++ / ScoreCAM on all alleles of specific model
    Output file in directory [output_dir]/[target_layer]/[cam_method]/[allele]:
    1. record.npy: CAM matrix
    2. [mhc/epitope]_heatmap.png: heatmap of selected measurements
    3. epitope_clustermap.png: clustermap of selected measurements, only for epitope
    '''
    parser = argparse.ArgumentParser(prog='run_cam', description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    data_args = parser.add_argument_group("Data Arguments")
    data_args.add_argument('--mhc_file', required=True, help='MHCI binding domain encoding file')
    data_args.add_argument('--dataframe_file', required=True, help='the dataframe file of peptides')
    data_args.add_argument('--dataset_file', required=True, help='the dataset file of peptides')
    data_args.add_argument('--predict_col', required=True, help='column name of the prediction')
    data_args.add_argument('--predict_threshold', required=True, help='threshold of the prediction')
    data_args.add_argument('--encode', required=True, help='onehot or blosum')
    
    model_args = parser.add_argument_group('Model Arguments')
    model_args.add_argument('--model_path', required=True, help='model architecture file')
    model_args.add_argument('--model_state_path', required=True, help='model state file')
    model_args.add_argument('--cat_dim', required=False, type=int, default=1, help='concatenating dimension of mhc and peptide')
    
    cam_args = parser.add_argument_group('CAM Arguments')
    cam_args.add_argument('--mhc_target_layers', required=False, default='', help='target layers of MHCModel, ex: 0,1,2')
    cam_args.add_argument('--epitope_target_layers', required=False, default='', help='target layers of EpitopeModel, ex:0,1,2')
    cam_args.add_argument('--pre_mhc_len', required=True, type=int)
    cam_args.add_argument('--pre_epitope_len', required=True, type=int)
    cam_args.add_argument('--post_mhc_len', required=False, default='', help='the length should be the same as mhc_target_layers')
    cam_args.add_argument('--post_epitope_len', required=False, default='', help='the length should be the same as epitope_target_layers')
    cam_args.add_argument('--cam_list', required=True, help='GradCAM,GradCAMpp,ScoreCAM')
    
    other_args = parser.add_argument_group('Other Arguments')
    other_args.add_argument('--outdir', required=True, help='the output directory')

    return parser


if __name__ == '__main__':
    """""""""""""""""""""""""""""""""""""""""
    # Arguments
    """""""""""""""""""""""""""""""""""""""""
    args = ArgumentParser().parse_args()

    # data
    mhc_dict = np.load(args.mhc_file, allow_pickle=True)[()]
    df = pd.read_csv(args.dataframe_file, index_col=0)
    dataset = torch.load(args.dataset_file)
    predict_col = args.predict_col
    predict_threshold = float(args.predict_threshold)
    if args.encode == 'onehot':
        dim = 21
    elif args.encode == 'blosum':
        dim = 24
    else:
        print("Wrong encoding method")
        raise ValueError

    # model
    module_path = '.'.join(re.split(r'[\/\.]', args.model_path)[:-1])
    module = importlib.import_module(module_path)
    model = module.CombineModel(module.MHCModel(dim), module.EpitopeModel(dim))
    model_state_dict = torch.load(args.model_state_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict["model_state_dict"])
    cat_dim = args.cat_dim
    if torch.cuda.is_available():
        cuda = True
    else:
        cuda = False

    # CAM
    pre_mhc_len = args.pre_mhc_len
    pre_epitope_len = args.pre_epitope_len
    post_mhc_len = [int(i) for i in args.post_mhc_len.split(',') if args.post_mhc_len != '']
    post_epitope_len = [int(i) for i in args.post_epitope_len.split(',') if args.post_epitope_len != '']
    cam_list = args.cam_list.split(',')

    # other
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)


    """""""""""""""""""""""""""""""""""""""""
    # CAM
    """""""""""""""""""""""""""""""""""""""""
    alleles = df['mhc'].unique()
    num = np.inf
    batch_size = 256

    for target in ['mhc', 'epitope']:
        print("Current target: ", target)

        # set target_layers and target_lens
        if target == 'mhc':
            target_model_layer = model.modelA.main
            target_layers = [int(i) for i in args.mhc_target_layers.split(',') if args.mhc_target_layers != '']
            target_mhc_lens = post_mhc_len
            target_epitope_lens = [0]*len(target_mhc_lens)
        else:
            target_model_layer = model.modelB.main
            target_layers = [int(i) for i in args.epitope_target_layers.split(',') if args.epitope_target_layers != '']
            target_epitope_lens = post_epitope_len
            target_mhc_lens = [0]*len(target_epitope_lens)

        # run CAM per target_layer
        for i in range(len(target_layers)):
            target_layer = target_layers[i]
            print("Current target layer: ", target_layer)

            output_dir = '%s/%s_%d'%(args.outdir, target, target_layer)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            len_tuple = (pre_mhc_len, target_mhc_lens[i], pre_epitope_len, target_epitope_lens[i])

            cam_dict = dict()
            for cam in cam_list:
                if cam == 'GradCAM':
                    gradcam = GradCAM(model, target_model_layer[target_layer], len_tuple, cuda, cat_dim)
                    cam_dict[cam] = gradcam
                elif cam == 'GradCAMpp':
                    gradcampp = GradCAMpp(model, target_model_layer[target_layer], len_tuple, cuda, cat_dim)
                    cam_dict[cam] = gradcampp
                elif cam == 'ScoreCAM':
                    scorecam = ScoreCAM(model, target_model_layer[target_layer], len_tuple, cuda, cat_dim)
                    cam_dict[cam] = scorecam
                else:
                    print('CAM method error: ', cam)

            # per CAM method
            for name, cam in cam_dict.items():
                print("Current CAM: ", name)

                if not os.path.isdir('%s/%s'%(output_dir, name)):
                    os.mkdir('%s/%s'%(output_dir, name))

                masks = Masks(df, dataset, mhc_dict, cam)

                # per allele
                for allele in tqdm(alleles):
                    allele_name = allele[0] + allele[2:4] + allele[5:7]
                    masks(allele, predict_col, predict_threshold, num, batch_size, "%s/%s/%s"%(output_dir, name, allele_name))