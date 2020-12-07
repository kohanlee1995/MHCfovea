import os, sys, re, subprocess, argparse
import numpy as np
import pandas as pd


def AminoAcidReplacement(seq):
    aa_str = 'ACDEFGHIKLMNPQRSTVWY'
    seq_list = list(seq)
    for i in range(len(seq_list)):
        if seq_list[i] not in aa_str:
            seq_list[i] = 'X'
    return ''.join(seq_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='run_mhcflurry')
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--model_dir', required=True)
    
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    model_dir = args.model_dir
    tmp_file = '/tmp/run_mhcflurry_tmp.csv'

    df = pd.read_csv(input_file, index_col=0)
    df['sequence'] = df['sequence'].apply(lambda x: AminoAcidReplacement(x))
    df = df.rename(columns={'mhc':'allele', 'sequence':'peptide'})
    df[['allele', 'peptide']].to_csv(tmp_file)

    command = ['mhcflurry-predict', tmp_file]
    command += ['--models', model_dir]
    command += ['--out', output_file]
    print(command)
    process = subprocess.call(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("Complete")