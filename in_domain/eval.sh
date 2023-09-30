#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env_name>

# Benchmark
export DETECTRON2_DATASETS="datasets"
TEST_DATASETS="ade20k_sem_seg_val ade20k_full_sem_seg_val pascal_context_59_sem_seg_val pascal_context_459_sem_seg_val voc_2012_sem_seg_val_bg"

# Run experiments
for DATASET in $TEST_DATASETS
do
 python train_net.py --num-gpus 1 --eval-only --config-file configs/<config_file>.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS weights/<model_weights>.pth OUTPUT_DIR output/<model_name>/$DATASET
done

# Combine results
python mess/evaluation/mess_evaluation.py --model_outputs output/<model_name> output/<model2_name> <...>

# Run evaluation with:
# nohup bash in_domain/eval.sh > in_domain_eval.log &
# tail -f in_domain_eval.log
