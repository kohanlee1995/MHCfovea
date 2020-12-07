import os, sys, re, subprocess, argparse
import numpy as np
import pandas as pd


def SplitFile(df, num, dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    for allele, sub_df in df.groupby('mhc'):
        allele_name = _convert_allele(allele)
        size = sub_df.shape[0]
        
        if not os.path.isdir('%s/%s'%(dirname, allele_name)):
            os.mkdir('%s/%s'%(dirname, allele_name))
        
        for i in range(int(np.ceil(size/num))):
            filename = '%s/%s/%s_%d.tsv'%(dirname, allele_name, allele_name, i+1)
            sub_df.iloc[num*i: num*(i+1)][['sequence', 'bind', 'mhc']].to_csv(filename, sep='\t', header=False, index=False)


def Prediction(alleles, input_dir, output_dir, netmhcpan_path):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    for allele in alleles:
        if not os.path.isdir("%s/%s"%(output_dir, allele)):
            os.mkdir("%s/%s"%(output_dir, allele))

        for file in os.listdir("%s/%s"%(input_dir, allele)):
            if os.path.isfile("%s/%s/%s"%(output_dir, allele, file)):
                print("%s already exists"%file)
                continue

            print("%s processing..."%file)

            command = [netmhcpan_path, "-p"]
            command += ["-a", allele]
            command += ["-f", "%s/%s/%s"%(input_dir, allele, file)]
            command += ["-xls", "-xlsfile", "%s/%s/%s"%(output_dir, allele, file)]
            print(command)

            process = subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            print("%s complete"%file)


def MergeFile(df, dirname):
    df['netmhcpan'] = 0
    for allele in os.listdir(dirname):
        preds = list()
        file_num = len(os.listdir('%s/%s'%(dirname, allele)))
        for i in range(file_num):
            temp_df = pd.read_csv('%s/%s/%s_%d.tsv'%(dirname, allele, allele, i+1), header=1, sep='\t')
            preds += list(temp_df['EL-score'])
        df.loc[df['mhc']=='%s*%s:%s'%(allele[4], allele[5:7], allele[7:9]), 'netmhcpan'] = preds
    return df


def _convert_allele(allele):
    if re.match(r'[ABC]\*[0-9]+\:[0-9]+', allele):
        split = re.split(r'[\*\:]', allele)
        return 'HLA-%s%s%s'%(split[0], split[1], split[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='run_netmhcpan')
    parser.add_argument('--input_df', required=True)
    parser.add_argument('--output_df', required=True)
    parser.add_argument('--pre_run_dir', required=True)
    parser.add_argument('--post_run_dir', required=True)
    parser.add_argument('--num_per_file', required=False, default=1000, type=int)
    parser.add_argument('--netmhcpan_path', required=True)
    parser.add_argument('--function', required=True, help='SplitFile/Prediction/MergeFile')
    parser.add_argument('--alleles', required=False, default='', help='HLA-A0202,HLA-A0201,...')

    args = parser.parse_args()
    function = args.function

    if function == 'SplitFile':
        df = pd.read_csv(args.input_df, index_col=0)
        num = args.num_per_file
        dirname = args.pre_run_dir
        SplitFile(df, num, dirname)

    elif function == 'Prediction':
        input_dir = args.pre_run_dir
        output_dir = args.post_run_dir
        netmhcpan_path = args.netmhcpan_path
        alleles = args.alleles.split(',')
        if alleles == ['']:
            alleles = os.listdir(input_dir)
        Prediction(alleles, input_dir, output_dir, netmhcpan_path)

    elif function == 'MergeFile':
        df = pd.read_csv(args.input_df, index_col=0)
        dirname = args.post_run_dir
        df = MergeFile(df, dirname)
        df.to_csv(args.output_df)

    else:
        print('No %s function'%function)
        raise ValueError