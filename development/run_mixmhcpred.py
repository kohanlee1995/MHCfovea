import os, sys, re, subprocess, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def SplitFile(df, dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    for allele, sub_df in df.groupby('mhc'):
        allele_name = _convert_allele(allele)
        with open('%s/%s.fa'%(dirname, allele_name), 'w') as f:
            for i, seq in enumerate(sub_df['sequence']):
                if len(seq) == 15:
                    continue
                if not re.match(r'[ACDEFGHIKLMNPQRSTVWY]+$', seq):
                    continue
                f.write('>peptide_%d\n'%i)
                f.write('%s\n'%seq)


def Prediction(input_dir, output_dir, mixmhcpred_path):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    alleles = os.listdir(input_dir)
    alleles = [i.split('.')[0] for i in alleles]

    for allele in alleles:
        command = [mixmhcpred_path]
        command += ['-i', '%s/%s.fa'%(input_dir, allele)]
        command += ['-o', '%s/%s.txt'%(output_dir, allele)]
        command += ['-a', allele]
        print(command)

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process.communicate()
        process.wait()

        print('%s complete'%allele)


def MergeFile(df, dirname):
    df['mixmhcpred'] = np.nan
    for file in tqdm(os.listdir(dirname)):
        allele = file.split('.')[0]
        temp_df = pd.read_csv('%s/%s'%(dirname, file), header=11, sep='\t')
        preds = list(temp_df['Score_%s'%allele])
        df.loc[(df['mhc']=='%s*%s:%s'%(allele[0], allele[1:3], allele[3:5])) &
               (df['peptide_length'] < 15) &
               (df['sequence'].str.match(r'[ACDEFGHIKLMNPQRSTVWY]+$')), 'mixmhcpred'] = preds
    return df


def _convert_allele(allele):
    if re.match(r'[ABC]\*[0-9]+\:[0-9]+', allele):
        split = re.split(r'[\*\:]', allele)
        return '%s%s%s'%(split[0], split[1], split[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='run_mixmhcpred')
    parser.add_argument('--input_df', required=True)
    parser.add_argument('--output_df', required=True)
    parser.add_argument('--pre_run_dir', required=True)
    parser.add_argument('--post_run_dir', required=True)
    parser.add_argument('--mixmhcpred_path', required=True)
    parser.add_argument('--function', required=True, help='SplitFile/Prediction/MergeFile')

    args = parser.parse_args()
    function = args.function

    if function == 'SplitFile':
        df = pd.read_csv(args.input_df, index_col=0)
        dirname = args.pre_run_dir
        SplitFile(df, dirname)

    elif function == 'Prediction':
        input_dir = args.pre_run_dir
        output_dir = args.post_run_dir
        mixmhcpred_path = args.mixmhcpred_path
        Prediction(input_dir, output_dir, mixmhcpred_path)

    elif function == 'MergeFile':
        df = pd.read_csv(args.input_df, index_col=0)
        dirname = args.post_run_dir
        df = MergeFile(df, dirname)
        df.to_csv(args.output_df)

    else:
        print('No %s function'%function)
        raise ValueError
