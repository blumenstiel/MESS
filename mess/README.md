# Multi-domain Evaluation of Semantic Segmentation (MESS)

[[Website](https://github.io)] [[arXiv](https://arxiv.org/)] [[GitHub](https://github.com/blumenstiel/MESS)]

This directory contains the code for the MESS benchmark. The benchmark covers 22 datasets and is currently designed for zero-shot semantic segmentation.

## Usage

Place this `mess` directory to your project, and follow the steps in [DATASETS.md](mess/DATASETS.md) for downloading and preparing the datasets.

You can register the datasets to Detectron2 by adding `import mess.datasets` to your evaluation code.

For evaluating all datasets with a Detectron2 model, you can use the following script:
```sh
conda activate <your_env>

TEST_DATASETS="mhp_v1_sem_seg_test foodseg103_sem_seg_test bdd100k_sem_seg_val dark_zurich_sem_seg_val atlantis_sem_seg_test dram_sem_seg_test isaid_sem_seg_val isprs_potsdam_sem_seg_test_irrg worldfloods_sem_seg_test_irrg floodnet_sem_seg_test uavid_sem_seg_val kvasir_instrument_sem_seg_test chase_db1_sem_seg_test cryonuseg_sem_seg_test paxray_sem_seg_test_lungs paxray_sem_seg_test_bones paxray_sem_seg_test_mediastinum paxray_sem_seg_test_diaphragm corrosion_cs_sem_seg_test deepcrack_sem_seg_test pst900_sem_seg_test zerowaste_sem_seg_test suim_sem_seg_test cub_200_sem_seg_test cwfid_sem_seg_test"

for DATASET in $TEST_DATASETS
do
 python evaluate.py --eval-only --config-file <your_config>.yaml --num-gpus 1 OUTPUT_DIR output/$DATASET DATASETS.TEST \(\"$DATASET\",\)
done
```

We also provide an adapted evaluator class `MESSSemSegEvaluator` in `mess.evaluation` and scripts to use the datasets with MMSegmentation and Torchvision. Note that you still have to install Detectron2 and register the datasets.

```python
# MMSegmentation 
# Replace build_dataset (from mmseg.datasets) with
import mess.datasets
from mess.datasets.MMSegDataset import build_mmseg_dataset
dataset = build_mmseg_dataset(cfg)
# Select the dataset with: cfg['type'] = '<dataset_name>'

# Torchvision
import mess.datasets
from mess.datasets.TorchvisionDataset import TorchvisionDataset
dataset = TorchvisionDataset('<dataset_name>', transform, mask_transform)
```
