import os, json, argparse, subprocess, warnings
warnings.filterwarnings('ignore')


def ArgumentParser():
    description = '''
    Prediction on alleles
    Use predictor.py to predict binding affinity of each allele within an allele group (ex. A*01).
    If mhc_alleles_file is not available, all alleles in the allele group will be predicted.
    
    Output:
    Create a directory for each allele group
    [output_dir]/[allele_group] contains:
    1. prediction.csv: with new column "score" for specific mode or [allele] for general mode
    2. motif.npy: dictionary with allele as key and motif array as value (number of positive samples >= 10)
    3. record.json: recording arguments
    '''
    parser = argparse.ArgumentParser(prog='run_pan_allele', description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    data_args = parser.add_argument_group('Data Arguments')
    data_args.add_argument('--mhc_encode_file', required=True, help='MHCI sequence encoding file')
    data_args.add_argument('--mhc_alleles_file', required=True, help='a json file contains allele groups to alleles dict')
    data_args.add_argument('--mhc_allele_groups', required=True, help='allele groups, ex. A*01,A*02')
    data_args.add_argument('--peptide_dataframe', required=True, help='csv file, contains "sequence" and "mhc" columns')
    data_args.add_argument('--peptide_dataset', required=False, default=None, help='dataset file built from the "BuildDataset" function in "util.py", default=None')
    data_args.add_argument('--encoding_method', required=False, type=str, default='onehot', help='onehot or blosum, default=onehot')
    
    model_args = parser.add_argument_group('Model Arguments')
    model_args.add_argument('--model_file', required=True, help='model architecture file from the same directory')
    model_args.add_argument('--model_state_dir', required=True, help='model state directory')

    other_args = parser.add_argument_group('Other Arguments')
    other_args.add_argument('--output_dir', required=True, help='output directory')
    other_args.add_argument('--seqlogo_threshold', required=False, type=float, default=0.9, help='prediction threshold for building seqlogo dataframe, default=0.9')
    
    return parser


if __name__ == "__main__":
    """""""""""""""""""""""""""""""""""""""""
    # Arguments
    """""""""""""""""""""""""""""""""""""""""
    args = ArgumentParser().parse_args()

    # data
    mhc_encode_file = args.mhc_encode_file
    mhc_alleles = json.load(open(args.mhc_alleles_file, 'r'))
    mhc_allele_groups = args.mhc_allele_groups.split(',')
    peptide_dataframe = args.peptide_dataframe
    peptide_dataset = args.peptide_dataset
    encoding_method = args.encoding_method
    
    # model
    model_file = args.model_file
    model_state_dir = args.model_state_dir
    
    # others
    output_dir = args.output_dir
    seqlogo_threshold = args.seqlogo_threshold

    # mkdir output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    """""""""""""""""""""""""""""""""""""""""
    # Main
    """""""""""""""""""""""""""""""""""""""""
    for allele_group in mhc_allele_groups:
        alleles = mhc_alleles[allele_group]
        
        command = ['python3', 'predictor.py']
        command += ['--mhc_encode_file', mhc_encode_file]
        command += ['--peptide_dataframe', peptide_dataframe]
        command += ['--peptide_dataset', peptide_dataset]
        command += ['--encoding_method', encoding_method]
        command += ['--model_file', model_file]
        command += ['--model_state_dir', model_state_dir]
        command += ['--output_dir', '%s/%s%s'%(output_dir, allele_group[0], allele_group[2:])]
        command += ['--seqlogo_threshold', str(seqlogo_threshold)]
        command += ['--alleles', ','.join(alleles)]

        print(command)

        process = subprocess.call(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        print("%s Complete"%allele_group)