import os, re, json, sys, random, copy, argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from util import *
import warnings
warnings.filterwarnings('ignore')


# remove duplicates from df_a
def RemoveDuplicates(df_a, df_b, cols):
    df = pd.concat([df_a, df_b[cols], df_b[cols]]).drop_duplicates(subset=cols, keep=False, ignore_index=True)
    return df


# preprocessing for IEDB MS data
def MSPreprocess(filename, min_peptide_length, max_peptide_length):
    df = pd.read_csv(filename)

    # rename columns
    rename_columns = ["%s - %s"%(i.split(".")[0], df.iloc[0][i]) for i in df.columns]
    df = df.rename(columns={df.columns[i]: rename_columns[i] for i in range(len(rename_columns))})
    df = df.drop(0)

    # filter
    df = df[["Reference - MHC ligand ID",
             "Reference - PubMed ID",
             "Epitope - Description",
             "Epitope - Starting Position",
             "Epitope - Ending Position",
             "Epitope - Parent Protein Accession",
             "Antigen Processing Cells - Cell Tissue Type",
             "Antigen Processing Cells - Cell Type",
             "MHC - Allele Name",
             "Host - MHC Types Present"]]
    # epitope length
    df["Epitope - Length"] = df["Epitope - Description"].apply(lambda x: len(x))
    df = df[(df["Epitope - Length"] >= min_peptide_length) & (df["Epitope - Length"] <= max_peptide_length)]
    # mono-allelic
    df = df[df["MHC - Allele Name"].str.match(r'^HLA-[A/B/C]\*\d+\:\d+$')]
    df["MHC - Allele Name"] = df["MHC - Allele Name"].apply(lambda x: x.replace("HLA-",""))

    return df


# preprocessing for IEDB assay data
def AssayPreprocess(filename, species, min_peptide_length, max_peptide_length):
    df = pd.read_csv(filename, sep='\t')
    df = df[df["species"] == species]
    df = df[df["mhc"].str.contains("HLA-[ABC]\*\d+\:\d+")]
    df["mhc"] = df["mhc"].apply(lambda x: x.replace("HLA-",""))
    df = df[(df["peptide_length"] >= min_peptide_length) & (df["peptide_length"] <= max_peptide_length)]
    df["value"] = df["meas"].apply(lambda x: max(1 - np.log10(x)/np.log10(50000), 0))
    df["bind"] = (df["meas"] <= 500).astype(int)
    df["source"] = "assay"
    return df


# build hit dataframe
def BuildHit(df):
    hit_df = df[[
        "MHC - Allele Name",
        "Epitope - Parent Protein Accession",
        "Epitope - Starting Position",
        "Epitope - Length",
        "Epitope - Description"
    ]]
    hit_df = hit_df.rename(columns={
        "MHC - Allele Name": "mhc",
        "Epitope - Parent Protein Accession": "protein",
        "Epitope - Starting Position": "start_pos",
        "Epitope - Length": "peptide_length",
        "Epitope - Description": "sequence"
    })
    hit_df["meas"] = 1
    hit_df["value"] = 1
    hit_df["bind"] = 1
    hit_df["source"] = "MS"
    return hit_df


# build decoy from the same protein of the hit sample
def BuildProtDecoy(prot_dict, prot_len_dict, df, len_dict):
    decoy_list = list()
    alleles = list(df['mhc'].unique())
    
    for allele in tqdm(alleles):
        temp_df = df[(df['mhc'] == allele) & (df['bind'] == 1)]
        prots = list(temp_df['protein'].unique())
        
        for prot in prots:
            pos_num = temp_df[temp_df['protein'] == prot].shape[0]
            start_pos_list = list(temp_df[temp_df['protein'] == prot]['start_pos'].unique())
            
            for length, multiple in len_dict.items():
                decoy_num = multiple * pos_num
                try:
                    candidate_pos = [i for i in range(prot_len_dict[prot] - length)
                                     if i not in start_pos_list]
                except:
                    continue
                candidate_pos = random.sample(candidate_pos, min(len(candidate_pos), decoy_num))
                
                for pos in candidate_pos:
                    d = {'mhc': allele,
                         'protein': prot,
                         'start_pos': pos,
                         'peptide_length': length,
                         'sequence': prot_dict[prot][pos: pos+length]}
                    decoy_list.append(d)
            
    decoy_df = pd.DataFrame(decoy_list)
    decoy_df = decoy_df.drop_duplicates(ignore_index=True)
    decoy_df["meas"] = 50000
    decoy_df["value"] = 0
    decoy_df['bind'] = 0
    decoy_df['source'] = 'protein_decoy'
    
    return decoy_df


# build decoy from random peptides
def BuildRandomDecoy(prot_dict, prot_len_dict, df, len_dict):
    decoy_list = list()
    prot_list = list(prot_dict.keys())
    alleles = list(df['mhc'].unique())
    
    for allele in tqdm(alleles):
        pos_num = df.loc[(df['mhc'] == allele) & (df['bind'] == 1)].shape[0]
        
        for length, multiple in len_dict.items():
            decoy_num = multiple * pos_num
            
            for i in range(decoy_num):
                choose = False
                while not choose:
                    prot_id = random.choice(prot_list)
                    try:
                        start_pos = random.choice(range(prot_len_dict[prot_id]-length))
                        choose = True
                    except:
                        choose = False
                
                d = {'mhc': allele,
                     'protein': prot_id,
                     'start_pos': start_pos,
                     'peptide_length': length,
                     'sequence': prot_dict[prot_id][start_pos: start_pos+length]}
                
                decoy_list.append(d)
    
    decoy_df = pd.DataFrame(decoy_list)
    decoy_df = decoy_df.drop_duplicates(ignore_index=True)
    decoy_df["meas"] = 50000
    decoy_df["value"] = 0
    decoy_df['bind'] = 0
    decoy_df['source'] = 'random_decoy'
    
    return decoy_df


def ArgumentParser():
    description = """
    Build dataset for downsampling model
    Input contains:
    1. MS data (.csv)
    2. Assay data (.txt)
    3. UniProt data (.json)

    Arguments:
    1. train_valid_prop
    2. random_seed
    3. encoding_method: OneHot or Blosum
    4. train_decoy_ratio
    5. test_decoy_ratio

    Output:
    1. dataframe/[train_hit / train_decoy_n / valid / test].csv
    2. [encoding_method]/[train_hit / train_decoy_n / valid / test].pt
        the shape of x: (encoding size, 15(epitope length))
        the shape of y: (3,) ([index, classification value, regression value])
    """
    parser = argparse.ArgumentParser(prog="BuildDatasetDownsampling", description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ms_data', required=True, help='the csv filename of MS data')
    parser.add_argument('--assay_data', required=True, help='the txt filename of assay data')
    parser.add_argument('--uniprot', required=True, help='the json filename of UniProt data')
    parser.add_argument('--train_valid_prop', required=False, type=float, default=0.95, help='default = 0.95')
    parser.add_argument('--random_seed', required=False, type=int, default=0, help='default = 0')
    parser.add_argument('--encoding_method', required=True, help='onehot or blosum')
    parser.add_argument('--train_decoy_times', required=True, type=int, help='decoy size of each length / hit size of training')
    parser.add_argument('--test_decoy_times', required=True, type=int, help='decoy size of each length / hit size of testing')
    parser.add_argument('--output_dir', required=True, help='the dirname of output')

    return parser


if __name__ == "__main__":
    """""""""""""""""""""""""""""""""
    # Loading data and arguments
    """""""""""""""""""""""""""""""""
    print("Loading data and arguemnts...")

    min_peptide_length, max_peptide_length = 8, 15

    args = ArgumentParser().parse_args()

    # IEDB data
    ms_df = MSPreprocess(args.ms_data, min_peptide_length, max_peptide_length)
    assay_df = AssayPreprocess(args.assay_data, "human", min_peptide_length, max_peptide_length)
    
    # UniProt data
    uniprot_dict = json.load(open(args.uniprot, 'r'))
    uniprot_len_dict = dict()
    for k, v in uniprot_dict.items():
        uniprot_len_dict[k] = len(v)

    # Basic arguments
    unique_columns = ["mhc", "sequence"] # for removing duplicates
    test_ref_id = 31844290
    train_valid_prop = args.train_valid_prop
    random_seed = args.random_seed
    encoding_method = args.encoding_method
    
    # Decoy arguments
    train_decoy_times = args.train_decoy_times
    test_decoy_times = args.test_decoy_times
    prot_decoy_len_dict = dict({i:2 for i in range(8, 16)})
    train_random_decoy_len_dict = dict({i: train_decoy_times for i in range(8, 16)})
    test_random_decoy_len_dict = dict({i: test_decoy_times for i in range(8, 16)})

    # output directory
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir('%s/dataframe'%output_dir):
        os.mkdir('%s/dataframe'%output_dir)
    if not os.path.isdir('%s/%s'%(output_dir, encoding_method)):
        os.mkdir('%s/%s'%(output_dir, encoding_method))

    
    """""""""""""""""""""""""""""""""
    # Build MS positive data
    """""""""""""""""""""""""""""""""
    print("Build MS positive data...")

    # split test df
    test_df = ms_df[ms_df["Reference - PubMed ID"] == test_ref_id]
    non_test_df = ms_df.drop(test_df.index)

    # df preprocess
    test_df = BuildHit(test_df)
    non_test_df = BuildHit(non_test_df)

    # drop duplicates
    non_test_df = non_test_df.drop_duplicates(subset=unique_columns, ignore_index=True)
    test_df = test_df.drop_duplicates(subset=unique_columns, ignore_index=True)

    # test_df = test_df - non_test_df
    test_df = RemoveDuplicates(test_df, non_test_df, unique_columns)


    """""""""""""""""""""""""""""""""
    # Build MS decoy data
    """""""""""""""""""""""""""""""""
    print("Build MS decoy data...")

    test_random_decoy_df = BuildRandomDecoy(uniprot_dict, uniprot_len_dict, test_df, test_random_decoy_len_dict)
    test_prot_decoy_df = BuildProtDecoy(uniprot_dict, uniprot_len_dict, test_df, prot_decoy_len_dict)
    non_test_random_decoy_df = BuildRandomDecoy(uniprot_dict, uniprot_len_dict, non_test_df, train_random_decoy_len_dict)
    non_test_prot_decoy_df = BuildProtDecoy(uniprot_dict, uniprot_len_dict, non_test_df, prot_decoy_len_dict)

    test_decoy_df = pd.concat([test_random_decoy_df, test_prot_decoy_df]).drop_duplicates(ignore_index=True)
    non_test_decoy_df = pd.concat([non_test_random_decoy_df, non_test_prot_decoy_df]).drop_duplicates(ignore_index=True)

    # test_decoy_df = test_decoy_df - test_df - non_test_df
    print("Remove duplicates of test_decoy_df...")
    print("before removing, data size = ", test_decoy_df.shape)
    test_decoy_df = RemoveDuplicates(test_decoy_df, test_df, unique_columns)
    test_decoy_df = RemoveDuplicates(test_decoy_df, non_test_df, unique_columns)
    print("after removing, data size = ", test_decoy_df.shape)

    # non_test_decoy_df = non_test_decoy_df - test_df - non_test_df - test_decoy_df
    print("Remove duplicates of non_test_decoy_df...")
    print("before removing, data size = ", non_test_decoy_df.shape)
    non_test_decoy_df = RemoveDuplicates(non_test_decoy_df, test_df, unique_columns)
    non_test_decoy_df = RemoveDuplicates(non_test_decoy_df, non_test_df, unique_columns)
    non_test_decoy_df = RemoveDuplicates(non_test_decoy_df, test_decoy_df, unique_columns)
    print("after removing, data size = ", non_test_decoy_df.shape)


    """""""""""""""""""""""""""""""""
    # Split training and validation data
    """""""""""""""""""""""""""""""""
    print("Split training and validation data...")

    # MS data
    train_df = non_test_df.sample(frac=train_valid_prop, random_state=random_seed)
    valid_df = non_test_df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    train_decoy_df = non_test_decoy_df.sample(frac=train_valid_prop, random_state=random_seed)
    valid_decoy_df = non_test_decoy_df.drop(train_decoy_df.index).reset_index(drop=True)
    train_decoy_df = train_decoy_df.reset_index(drop=True)

    # assay data
    assay_train_df = assay_df.sample(frac=train_valid_prop, random_state=random_seed)
    assay_valid_df = assay_df.drop(assay_train_df.index).reset_index(drop=True)
    assay_train_df = assay_train_df.reset_index(drop=True)


    """""""""""""""""""""""""""""""""
    # Save dataframe and dataset
    """""""""""""""""""""""""""""""""
    print("Save dataframe and dataset...")

    common_columns = ["sequence", "peptide_length", "mhc", "meas", "value", "bind", "source"]

    # train_hit (MS + assay)
    print("Current: train_hit...")
    train_hit_df = pd.concat([train_df[common_columns], assay_train_df[common_columns]], ignore_index=True)
    train_hit_df.to_csv('%s/dataframe/train_hit.csv'%output_dir)
    train_hit_num = train_hit_df.shape[0]
    dataset = BuildDataset(train_hit_df, encoding_method, max_peptide_length)
    torch.save(dataset, "%s/%s/train_hit.pt"%(output_dir, encoding_method))

    # train_decoy
    decoy_file_num = int(np.floor(train_decoy_df.shape[0] / train_df.shape[0]))
    decoy_idx = train_decoy_df.index.to_numpy()
    np.random.seed(random_seed)
    np.random.shuffle(decoy_idx)
    decoy_idx_list = np.array_split(decoy_idx, decoy_file_num)

    print("Number of decoy file: ", decoy_file_num)
    for i in range(decoy_file_num):
        print("Current: train_decoy_%d..."%(i+1))
        # make sure the decoy index within the number limitation(16777216) of pytorch tensor
        # decoy_1 and decoy_21 have the same starting index
        if i % 20 == 0:
            start_idx = train_hit_num
        temp_decoy_df = train_decoy_df.loc[decoy_idx_list[i], common_columns]
        temp_decoy_df = temp_decoy_df.set_index(pd.Series(range(start_idx, temp_decoy_df.shape[0]+start_idx)))
        start_idx += temp_decoy_df.shape[0]
        temp_decoy_df.to_csv('%s/dataframe/train_decoy_%d.csv'%(output_dir, i+1))
        dataset = BuildDataset(temp_decoy_df, encoding_method, max_peptide_length)
        torch.save(dataset, '%s/%s/train_decoy_%d.pt'%(output_dir, encoding_method, i+1))

    # validation
    print("Current: valid...")
    valid_df = pd.concat([valid_df[common_columns], valid_decoy_df[common_columns], assay_valid_df[common_columns]], ignore_index=True)
    valid_df.to_csv('%s/dataframe/valid.csv'%output_dir)
    dataset = BuildDataset(valid_df, encoding_method, max_peptide_length)
    torch.save(dataset, "%s/%s/valid.pt"%(output_dir, encoding_method))

    # testing
    print("Current: test...")
    test_df = pd.concat([test_df[common_columns], test_decoy_df[common_columns]], ignore_index=True)
    test_df.to_csv('%s/dataframe/test.csv'%output_dir)
    dataset = BuildDataset(test_df, encoding_method, max_peptide_length)
    torch.save(dataset, "%s/%s/test.pt"%(output_dir, encoding_method))


    """""""""""""""""""""""""""""""""
    # Record
    """""""""""""""""""""""""""""""""
    with open('%s/record.txt'%output_dir, 'w') as f:
        print("TRAINING HIT DATA", file=f)
        print("MS data number: ", (train_hit_df['source']=='MS').sum(), file=f)
        print("Assay data number: ", (train_hit_df['source']=='assay').sum(), file=f)
        print("\tPositive: ", ((train_hit_df['source']=='assay') & (train_hit_df['bind']==1)).sum(), file=f)
        print("\tNegative: ", ((train_hit_df['source']=='assay') & (train_hit_df['bind']==0)).sum(), file=f)
        print("=================================", file=f)
        print("TRAINING DECOY", file=f)
        print("Total decoy number: ", train_decoy_df.shape[0], file=f)
        print("File number: ", decoy_file_num, file=f)
        print("Average data number of each file: ", len(decoy_idx_list[0]), file=f)
        print("=================================", file=f)
        print("VALIDATION DATA", file=f)
        print("MS data number: ", (valid_df['source']=='MS').sum(), file=f)
        print("Decoy data number: ", (valid_df['source']=='protein_decoy').sum() + (valid_df['source']=='random_decoy').sum(), file=f)
        print("Assay data number: ", (valid_df['source']=='assay').sum(), file=f)
        print("\tPositive: ", ((valid_df['source']=='assay') & (valid_df['bind']==1)).sum(), file=f)
        print("\tNegative: ", ((valid_df['source']=='assay') & (valid_df['bind']==0)).sum(), file=f)
        print("=================================", file=f)
        print("TESTING DATA", file=f)
        print("MS data number: ", (test_df['source']=='MS').sum(), file=f)
        print("Decoy data number: ", (test_df['source']=='protein_decoy').sum() + (test_df['source']=='random_decoy').sum(), file=f)