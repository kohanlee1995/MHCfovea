# Build Dataset

python3 build_dataset.py \
--ms_data ../data/IEDB/mhc_ligand_table_export_1593661050.csv \
--assay_data ../data/IEDB/bdata.20130222.mhci.txt \
--uniprot ../data/Uniprot/uniprot_seq.json \
--encoding_method onehot \
--train_decoy_times 10 \
--test_decoy_times 2 \
--output_dir ../data/raw

##########################################################################################

# Training
# decoy1: $i
# decoy5: $i,$((i+1)),$((i+2)),$((i+3)),$((i+4))
# decoy10: $i,$((i+1)),$((i+2)),$((i+3)),$((i+4)),$((i+5)),$((i+6)),$((i+7)),$((i+8)),$((i+9))
# decoy15: $i,$((i+1)),$((i+2)),$((i+3)),$((i+4)),$((i+5)),$((i+6)),$((i+7)),$((i+8)),$((i+9)),$((i+10)),$((i+11)),$((i+12)),$((i+13)),$((i+14))
# decoy30: $i,$((i+1)),$((i+2)),$((i+3)),$((i+4)),$((i+5)),$((i+6)),$((i+7)),$((i+8)),$((i+9)),$((i+10)),$((i+11)),$((i+12)),$((i+13)),$((i+14)),$((i+15)),$((i+16)),$((i+17)),$((i+18)),$((i+19)),$((i+20)),$((i+21)),$((i+22)),$((i+23)),$((i+24)),$((i+25)),$((i+26)),$((i+27)),$((i+28)),$((i+29))

export factor=1
for i in {};
do
    python3 trainer.py \
    --mhc_encode_file ../data/MHCI/MHCI_res182_onehot.npy \
    --dataframe_dir ../data/raw/dataframe \
    --dataset_dir ../data/raw/onehot \
    --decoy_num $i \
    --encoding_method onehot \
    --model_file model/res182_CNN_1.py \
    --method classification \
    --batch_size 32 \
    --num_epochs 30 \
    --optim_lr 1e-4 \
    --optim_weight_decay 1e-4 \
    --output_dir ../result/res182_decoy${factor}_CNN_1_1
done

##########################################################################################

# Testing

export data_name=valid
export model_state_dir=res182_decoy30_CNN_1_1
export model_num=3
python3 predictor.py \
--mhc_encode_file ../data/MHCI/MHCI_res182_onehot.npy \
--peptide_dataframe ../data/raw/dataframe/${data_name}.csv \
--peptide_dataset ../data/raw/onehot/${data_name}.pt \
--encoding_method onehot \
--model_file model/res182_CNN_1.py \
--model_state_dir ../result/${model_state_dir}/model_state \
--model_num ${model_num} \
--output_dir ../prediction/${data_name}/${model_state_dir}_${model_num} \
--seqlogo_threshold 0.9 \
--save_tmp \
--get_metrics

##########################################################################################

# netMHCpan4.1 prediction

python3 run_netmhcpan.py \
--input_df ../data/raw/dataframe/test.csv \
--output_df ../prediction/test/netmhcpan/prediction.csv \
--pre_run_dir ../prediction/test/netmhcpan/pre_run/ \
--post_run_dir ../prediction/test/netmhcpan/post_run/ \
--num_per_file 1000 \
--function SplitFile \
--netmhcpan_path /volume/immunotherapy/MHCIpredictor/netMHCpan4.1/netMHCpan-4.1/netMHCpan

python3 run_netmhcpan.py \
--input_df ../data/raw/dataframe/test.csv \
--output_df ../prediction/test/netmhcpan/prediction.csv \
--pre_run_dir ../prediction/test/netmhcpan/pre_run/ \
--post_run_dir ../prediction/test/netmhcpan/post_run/ \
--num_per_file 1000 \
--function Prediction \
--netmhcpan_path /volume/immunotherapy/MHCIpredictor/netMHCpan4.1/netMHCpan-4.1/netMHCpan \
--alleles 
"""
alleles
HLA-A0101,HLA-A0201,HLA-A0202,HLA-A0203,HLA-A0204,HLA-A0205,HLA-A0206,HLA-A0207,
HLA-A0211,HLA-A0301,HLA-A1101,HLA-A1102,HLA-A2301,HLA-A2402,HLA-A2407,HLA-A2501,
HLA-A2601,HLA-A2902,HLA-A3001,HLA-A3002,HLA-A3101,HLA-A3201,HLA-A3301,HLA-A3303,
HLA-A3401,HLA-A3402,HLA-A3601,HLA-A6601,HLA-A6801,HLA-A6802,HLA-A7401,HLA-B0702,
HLA-B0704,HLA-B0801,HLA-B1301,HLA-B1302,HLA-B1402,HLA-B1501,HLA-B1502,HLA-B1503,
HLA-B1510,HLA-B1517,HLA-B1801,HLA-B2705,HLA-B3501,HLA-B3503,HLA-B3507,HLA-B3701,
HLA-B3801,HLA-B3802,HLA-B4001,HLA-B4002,HLA-B4006,HLA-B4201,HLA-B4402,HLA-B4403,
HLA-B4501,HLA-B4601,HLA-B4901,HLA-B5001,HLA-B5101,HLA-B5201,HLA-B5301,HLA-B5401,
HLA-B5501,HLA-B5502,HLA-B5601,HLA-B5701,HLA-B5703,HLA-B5801,HLA-B5802,HLA-C0102,
HLA-C0202,HLA-C0302,HLA-C0303,HLA-C0304,HLA-C0401,HLA-C0403,HLA-C0501,HLA-C0602,
HLA-C0701,HLA-C0702,HLA-C0704,HLA-C0801,HLA-C0802,HLA-C1202,HLA-C1203,HLA-C1402,
HLA-C1403,HLA-C1502,HLA-C1601,HLA-C1701
"""

python3 run_netmhcpan.py \
--input_df ../data/raw/dataframe/test.csv \
--output_df ../prediction/test/netmhcpan/prediction.csv \
--pre_run_dir ../prediction/test/netmhcpan/pre_run/ \
--post_run_dir ../prediction/test/netmhcpan/post_run/ \
--num_per_file 1000 \
--function MergeFile \
--netmhcpan_path /volume/immunotherapy/MHCIpredictor/netMHCpan4.1/netMHCpan-4.1/netMHCpan

##########################################################################################

# MHCflurry2.0-variant prediction

python3 run_mhcflurry.py \
--input_file ../data/raw/dataframe/test.csv \
--output_file ../prediction/test/mhcflurry/prediction.csv \
--model_dir /volume/immunotherapy/MHCIpredictor/MHCflurry2.0/models/models_class1_pan_variants/models.no_additional_ms/

##########################################################################################

# MixMHCpred2.1 prediction

python3 run_mixmhcpred.py \
--input_df ../data/raw/dataframe/test.csv \
--output_df ../prediction/test/mixmhcpred/prediction.csv \
--pre_run_dir ../prediction/test/mixmhcpred/pre_run/ \
--post_run_dir ../prediction/test/mixmhcpred/post_run/ \
--function SplitFile \
--mixmhcpred_path /volume/immunotherapy/MHCIpredictor/MixMHCpred2.1/MixMHCpred

python3 run_mixmhcpred.py \
--input_df ../data/raw/dataframe/test.csv \
--output_df ../prediction/test/mixmhcpred/prediction.csv \
--pre_run_dir ../prediction/test/mixmhcpred/pre_run/ \
--post_run_dir ../prediction/test/mixmhcpred/post_run/ \
--function Prediction \
--mixmhcpred_path /volume/immunotherapy/MHCIpredictor/MixMHCpred2.1/MixMHCpred

python3 run_mixmhcpred.py \
--input_df ../data/raw/dataframe/test.csv \
--output_df ../prediction/test/mixmhcpred/prediction.csv \
--pre_run_dir ../prediction/test/mixmhcpred/pre_run/ \
--post_run_dir ../prediction/test/mixmhcpred/post_run/ \
--function MergeFile \
--mixmhcpred_path /volume/immunotherapy/MHCIpredictor/MixMHCpred2.1/MixMHCpred

##########################################################################################

# CAM

export data_name=train_hit
export model_state_dir=res182_decoy5_CNN_1_1
export model_num=18
for i in {51,56,61,66,71,76,81,86};
do
    python3 cam_run.py \
    --mhc_file ../data/MHCI/MHCI_res182_onehot.npy \
    --dataframe_file ../prediction/${data_name}/${model_state_dir}_${model_num}/tmp_prediction.csv \
    --dataset_file ../data/raw/onehot/${data_name}.pt \
    --predict_col decoy_${i} \
    --predict_threshold 0.9 \
    --encode onehot \
    --model_path model/res182_CNN_1.py \
    --model_state_path ../result/${model_state_dir}/model_state/decoy_${i}.tar \
    --mhc_target_layers 2 \
    --pre_mhc_len 182 \
    --pre_epitope_len 15 \
    --post_mhc_len 182 \
    --cam_list ScoreCAM \
    --outdir ../cam_result/${model_state_dir}_${data_name}/decoy_${i}
    echo "Decoy ${i} Complete"
    echo "======================================================"
done

##########################################################################################

# CAM interpretation

##########################################################################################

# MHCI prediction

export pred_dir="../prediction/pan_allele_2"
export mhc_alleles_file="${pred_dir}/selected_alleles.json"

for allele_group in {};
do
    echo "Current allele group: ${allele_group}"
    export alleles=$(jq --arg index $allele_group '.[$index]' $mhc_alleles_file | sed 's/"//g' | sed 's/\[//g' | sed 's/\]//g')
    export alleles=$(echo $alleles | sed 's/ //g')
    python3 predictor.py \
    --mhc_encode_file ../data/MHCI/MHCI_res182_onehot.npy \
    --peptide_dataframe ${pred_dir}/peptides.csv \
    --peptide_dataset ${pred_dir}/peptides.pt \
    --encoding_method onehot \
    --model_file model/res182_CNN_1.py \
    --model_state_dir ../result/res182_decoy5_CNN_1_1/model_state \
    --output_dir ${pred_dir}/output/$allele_group \
    --seqlogo_threshold 0.9 \
    --alleles ${alleles}
done

"""
allele_groups
A01,A02,A03,A11,A23,A24,A25,A26,
A29,A30,A31,A32,A33,A34,A36,A43,
A66,A68,A69,A74,A80,B07,B08,B13,
B14,B15,B18,B27,B35,B37,B38,B39,
B40,B41,B42,B44,B45,B46,B47,B48,
B49,B50,B51,B52,B53,B54,B55,B56,
B57,B58,B59,B67,B73,B78,B81,B82,
B83,C01,C02,C03,C04,C05,C06,C07,
C08,C12,C14,C15,C16,C17,C18
"""

##########################################################################################

# MHCI interpretation