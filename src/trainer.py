import os, re, json, sys, random, copy, importlib, argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from BA import BA
from util import *
import warnings
warnings.filterwarnings('ignore')


def ArgumentParser():
    description = '''
    MHCI-peptide binding prediction - training
    [Restriction]
    1. The length of peptide sequence must smaller than 15
    2. Single decoy file is recommended for balancing the label
    3. decoy_[n] and decoy_[n+20] which can't be used simultaneously have the same starting index
    4. GPU is recommended
    
    [Output]
    Major output directory:
    1. model_state: the directory to save best models
    2. decoy_[decoy_num]: the minor directory
    Minor output directory (decoy_[decoy_num]):
    1. config file for recording arguments
    2. log file for tensorboard
    3. model directory contains each epoch model state and the best one
    '''
    parser = argparse.ArgumentParser(prog='trainer', description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    data_args = parser.add_argument_group('Data Arguments')
    data_args.add_argument('--mhc_encode_file', required=True, help='MHCI binding domain encoding file')
    data_args.add_argument('--dataframe_dir', required=True, help='the directory of dataframe files, contains train_hit, train_decoy, valid')
    data_args.add_argument('--dataset_dir', required=True, help='the directory of dataset files, contains train_hit, train_decoy, valid')
    data_args.add_argument('--decoy_num', required=False, type=str, default='1', help='decoy number, default="1"')
    data_args.add_argument('--encoding_method', required=True, help='onehot or blosum')

    model_args = parser.add_argument_group('Model Arguments')
    model_args.add_argument('--model_file', required=True, help='model architecture file from the same directory')
    model_args.add_argument('--method', required=False, type=str, default='classification', help='classification or regression, default=classification')
    model_args.add_argument('--batch_size', required=False, type=int, default=32, help='batch size, default=32')
    model_args.add_argument('--num_epochs', required=False, type=int, default=30, help='number of epochs, default=30')
    model_args.add_argument('--optim_lr', required=False, type=float, default=1e-4, help='learning rate,default=1e-4')
    model_args.add_argument('--optim_weight_decay', required=False, type=float, default=1e-4, help='L2 regularization, default=1e-4')
    model_args.add_argument('--scheduler_milestones', required=False, type=str, default='0.5,0.8', help='the milestone of learning rate scheduler with gamma=0.1. Enter "0" to mute this function. Default="0.5,0.8"')

    other_args = parser.add_argument_group('Other Arguments')
    other_args.add_argument('--output_dir', required=True, help='the major output directory')

    return parser


if __name__ == "__main__":
    """""""""""""""""""""""""""""""""""""""""
    # Arguments
    """""""""""""""""""""""""""""""""""""""""
    args = ArgumentParser().parse_args()

    # data
    mhc_encode_file = args.mhc_encode_file
    dataframe_dir = args.dataframe_dir
    dataset_dir = args.dataset_dir
    decoy_num = list(map(int, args.decoy_num.split(',')))
    encoding_method = args.encoding_method

    # model
    model_file = args.model_file
    method = args.method
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    optim_lr = args.optim_lr
    optim_weight_decay = args.optim_weight_decay
    scheduler_milestones = list(map(float, args.scheduler_milestones.split(',')))

    # others
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir('%s/model_state'%output_dir):
        os.mkdir('%s/model_state'%output_dir)

    # default arguments
    valid_batch_size = 4096
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


    """""""""""""""""""""""""""""""""""""""""
    # Loading Data & Model
    """""""""""""""""""""""""""""""""""""""""
    print("Loading data and model...")

    # mhc encoding dict
    mhc_encode_dict = np.load(mhc_encode_file, allow_pickle=True)[()]

    # decoy files
    decoy_dataframe = ['%s/train_decoy_%s.csv'%(dataframe_dir, i) for i in decoy_num]
    decoy_dataset = ['%s/train_decoy_%s.pt'%(dataset_dir, i) for i in decoy_num]

    # training and validation data
    train_data = Data(
        ['%s/train_hit.csv'%dataframe_dir] + decoy_dataframe,
        ['%s/train_hit.pt'%dataset_dir] + decoy_dataset,
        batch_size,
        True)
    valid_data = Data(
        "%s/valid.csv"%dataframe_dir,
        "%s/valid.pt"%dataset_dir,
        valid_batch_size,
        False,
        len(decoy_dataframe)) # valid decoy times equal to the ratio of training data

    # model
    if encoding_method == 'onehot':
        dim = 21
    elif encoding_method == 'blosum':
        dim = 24
    else:
        print("Wrong encoding method")
        raise ValueError

    if method == "regression":
        criterion = nn.MSELoss()
        y_value_index = 2
    elif method == "classification":
        criterion = nn.BCELoss()
        y_value_index = 1
    else:
        print("Method must be 'regression' or 'classification'")
        raise ValueError

    module = importlib.import_module('.'.join(model_file.split('.')[0].split('/')))
    model = module.CombineModel(module.MHCModel(dim), module.EpitopeModel(dim))
    optimizer = optim.Adam(model.parameters(), lr=optim_lr, weight_decay=optim_weight_decay)


    """""""""""""""""""""""""""""""""""""""""
    # Training
    """""""""""""""""""""""""""""""""""""""""
    print("Training...")
    BA_model = BA(mhc_encode_dict, model, criterion, optimizer, scheduler_milestones, device, y_value_index, '%s/decoy_%d'%(output_dir, decoy_num[0]))
    BA_model.train(train_data.mhc_idx, train_data.dataloader, valid_data.mhc_idx, valid_data.dataloader, num_epochs)
    torch.save({'model_state_dict': BA_model.best_model.state_dict()}, "%s/model_state/decoy_%d.tar"%(output_dir, decoy_num[0]))


    """""""""""""""""""""""""""""""""""""""""
    # Recording
    """""""""""""""""""""""""""""""""""""""""
    record_dict = dict({
        'mhc_encode_file': mhc_encode_file,
        'dataframe_dir': dataframe_dir,
        'dataset_dir': dataset_dir,
        'decoy_num': decoy_num,
        'encoding_method': encoding_method,
        'model_file': model_file,
        'method': method,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'optim_lr': optim_lr,
        'optim_weight_decay': optim_weight_decay,
        'scheduler_milestones': scheduler_milestones
        })

    json.dump(record_dict, open('%s/decoy_%d/record.json'%(output_dir, decoy_num[0]), 'w'))
