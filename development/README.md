# Overview
This is a tutorial for reproducing all experiments in our research.
There are five steps, including:
1. Preparing the data
2. Training the model
3. Applying ScoreCAM
4. Summarizing the MHC-I-peptide connection
5. Analysis


# Step 1: Preparing the data
Prepare the training, decoy, validation, and benchmark datasets.
Several public datasets are needed, including:
1. ligand elution assay: extract from IEDB (https://www.iedb.org/)
2. binding assay: extract from IEDB (http://tools.iedb.org/static/main/binding_data_2013.zip)
3. UniProt (https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz)

```
python3 build_dataset.py \
    --ms_data ${LIGAND_ELUTION_DATA} \
    --assay_data ${BINDING_DATA} \
    --uniprot ${UNIPROT_DATA} \
    --encoding_method onehot \
    --train_decoy_times 10 \
    --test_decoy_times 2 \
    --output_dir ${DATA_DIR}
```

To reproduce MHCfovea, users can download the dataset from Mendeley Data (http://dx.doi.org/10.17632/c249p8gdzd.2). Then, use the **BuildDataset** function in the **util.py** to build pytorch dataset.


# Step 2: Training the model
Train the CNN-based model for binding prediction.
In this process, users will obtain 18 models as an ensemble model.
Essential data:
1. One-hot encoding of MHCI sequence: downloaded from Mendeley Data
2. Dataframe directory: contains csv files from step 1
3. Dataset directory: contains pytorch datasets from step 1

```
for i in {1,6,11,16,21,26,31,36,41,46,51,56,61,66,71,76,81,86};
do
    python3 trainer.py \
        --mhc_encode_file ${MHC_ENCODE_FILE} \
        --dataframe_dir ${DATA_DIR}/dataframe \
        --dataset_dir ${DATA_DIR}/onehot \
        --decoy_num $i,$((i+1)),$((i+2)),$((i+3)),$((i+4)) \
        --encoding_method onehot \
        --model_file model.py \
        --method classification \
        --batch_size 32 \
        --num_epochs 30 \
        --optim_lr 1e-4 \
        --optim_weight_decay 1e-4 \
        --output_dir ${TRAIN_RESUILT_DIR}
done
```


# Step 3: Applying ScoreCAM
Apply ScoreCAM on the models and training data to extract important positions of peptides or alleles

First, users have to predict the binding probability of experimental measurements of the training datset.
```
python3 predictor.py \
    --mhc_file ../data/MHCI_res182_seq.json \
    --rank_file ../data/score_rank.csv \
    --peptide_dataframe ${DATA_DIR}/dataframe/train_hit.csv \
    --peptide_dataset ${DATA_DIR}/onehot/train_hit.pt \
    --encoding_method onehot \
    --model_file model.py \
    --model_state_dir ${TRAIN_RESUILT_DIR}/model_state \
    --model_num 18 \
    --output_dir ${TRAIN_PREDICT_DIR} \
    --seqlogo_threshold 0.9 \
    --save_tmp \
    --get_metrics
```

Then, apply ScoreCAM on both peptide and allele parts.
```
for i in {1,6,11,16,21,26,31,36,41,46,51,56,61,66,71,76,81,86};
do
    python3 cam_run.py \
        --mhc_encode_file ${MHC_ENCODE_FILE} \
        --dataframe_file ${TRAIN_PREDICT_DIR}/tmp_prediction.csv \
        --dataset_file ${DATA_DIR}/onehot/train_hit.pt \
        --predict_col decoy_${i} \
        --predict_threshold 0.9 \
        --encode onehot \
        --model_path model.py \
        --model_state_path ${TRAIN_RESUILT_DIR}/model_state/decoy_${i}.tar \
        --mhc_target_layers 2 \
        --epitope_target_layers 0 \
        --pre_mhc_len 182 \
        --pre_epitope_len 15 \
        --post_mhc_len 182 \
        --post_epitope_len 15 \
        --cam_list ScoreCAM \
        --outdir ${CAM_RESULT_DIR}
    echo "Decoy ${i} Complete"
    echo "======================================================"
done
```


# Step 4: Summarizing the MHC-I-peptide connection
Summarize the connection between MHC-I alleles and binding peptides

First, predict the binding probability of all MHC-I alleles against 254,742 peptides (including all ligand elution data and some decoy peptides whose number was the same as ligand elution data of the benchmark dataset).

The peptide dataset and results can be downloaded from Mendeley Data. The pytorch dataset can be also built via the **BuildDataset** function in the **util.py**.

```
python3 run_pan_allele.py \
    --mhc_file ../data/MHCI_res182_seq.json \
    --rank_file ../data/score_rank.csv \
    --peptide_dataframe ${ALLELE_EXPANSION_DIR}/peptides.csv \
    --peptide_dataset ${ALLELE_EXPANSION_DIR}/peptides.pt \
    --encoding_method onehot \
    --model_file model.py \
    --model_state_dir ${TRAIN_RESUILT_DIR}/model_state \
    --output_dir ${ALLELE_EXPANSION_DIR} \
    --seqlogo_threshold 0.9 \
    --mhc_allele_groups ${ALLELE_GROUPS}
```


# Step 5: Analysis
## Data
Compare the training data between different predictors.
This is related to Supplementary Fig. 2 in the manuscript.
[Data analysis](Analysis-data.ipynb)

## Performance
Evaluate the performance of MHCfovea's predictor.
This is related to Fig. 2 and Supplementary Fig. 3 in the manuscript.
[Performance analysis](Analysis-performance.ipynb)

Evaluate the memorization of MHCfovea's predictor.
This is related to Supplementary Fig. 9 in the manuscript.
[Memorization analysis](Analysis-memorization.ipynb)

## ScoreCAM
Evalute the result of ScoreCAM applied in MHCfovea.
This is related to Fig. 3 and Supplementary Fig. 5 in the manuscript.
[ScoreCAM analysis](Analysis-ScoreCAM.ipynb)

## Summarization
Summarize the result of allele expansion to reveal the connection between alleles and peptides.
This is related to Fig. 4,5 and Supplementary Fig. 4,6,7,8 in the manuscript.
[Summarization analysis](Analysis-summarization.ipynb)

Analyze the HLA gorups with multi-cluster.
This is related to Fig. 6 in the manuscript.
[Multi-cluster analysis](Analysis-multicluster.ipynb)



