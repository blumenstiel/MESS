#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env_name>

# Benchmark
export DETECTRON2_DATASETS="datasets"
TEST_DATASETS="bdd100k_sem_seg_val dark_zurich_sem_seg_val mhp_v1_sem_seg_test foodseg103_sem_seg_test atlantis_sem_seg_test dram_sem_seg_test isaid_sem_seg_val isprs_potsdam_sem_seg_test_irrg worldfloods_sem_seg_test_irrg floodnet_sem_seg_test uavid_sem_seg_val kvasir_instrument_sem_seg_test chase_db1_sem_seg_test cryonuseg_sem_seg_test paxray_sem_seg_test_lungs paxray_sem_seg_test_bones paxray_sem_seg_test_mediastinum paxray_sem_seg_test_diaphragm corrosion_cs_sem_seg_test deepcrack_sem_seg_test pst900_sem_seg_test zerowaste_sem_seg_test suim_sem_seg_test cub_200_sem_seg_test cwfid_sem_seg_test"

# Run experiments
for DATASET in $TEST_DATASETS
do
 python train_net.py --num-gpus 1 --eval-only --config-file configs/<config_file>.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS weights/<model_weights>.pth OUTPUT_DIR output/<model_name>/$DATASET
done

# Combine results
python mess/evaluation/mess_evaluation.py --model_outputs output/<model_name> output/<model2_name> <...>

# Run evaluation with:
# nohup bash mess/eval.sh > eval.log &
# tail -f eval.log
