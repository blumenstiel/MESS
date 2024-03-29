# MESS – Multi-domain Evaluation of Semantic Segmentation

[[Website](https://blumenstiel.github.io/mess-benchmark/)] [[arXiv](https://arxiv.org/abs/2306.15521)] [[GitHub](https://github.com/blumenstiel/MESS)]

This directory contains the code for the MESS benchmark. The benchmark covers 22 datasets and is currently designed for zero-shot semantic segmentation.

## Usage

Place this `mess` directory to your project, and follow the steps in [DATASETS.md](DATASETS.md) for downloading and preparing the datasets.

You can register the datasets to detectron2 by adding `import mess.datasets` to your evaluation code.

For evaluating all datasets with a detectron2 model, you can use the following script:
```sh
conda activate <your_env>

TEST_DATASETS="mhp_v1_sem_seg_test foodseg103_sem_seg_test bdd100k_sem_seg_val dark_zurich_sem_seg_val atlantis_sem_seg_test dram_sem_seg_test isaid_sem_seg_val isprs_potsdam_sem_seg_test_irrg worldfloods_sem_seg_test_irrg floodnet_sem_seg_test uavid_sem_seg_val kvasir_instrument_sem_seg_test chase_db1_sem_seg_test cryonuseg_sem_seg_test paxray_sem_seg_test_lungs paxray_sem_seg_test_bones paxray_sem_seg_test_mediastinum paxray_sem_seg_test_diaphragm corrosion_cs_sem_seg_test deepcrack_sem_seg_test pst900_sem_seg_test zerowaste_sem_seg_test suim_sem_seg_test cub_200_sem_seg_test cwfid_sem_seg_test"

for DATASET in $TEST_DATASETS
do
 python evaluate.py --eval-only --config-file <your_config>.yaml --num-gpus 1 OUTPUT_DIR output/$DATASET DATASETS.TEST \(\"$DATASET\",\)
done
```

You can combine the results of the seperate datasets with the following script.
```sh
python mess/evaluation/mess_evaluation.py --model_outputs output/<model_name> output/<model2_name> <...>
# default values: --metrics [mIoU], --results_dir results/
```

We also provide an adapted evaluator class `MESSSemSegEvaluator` in `mess.evaluation` to calculate the mIoU for classes of interest (CoI-mIoU). Scripts to use the datasets with MMSegmentation and Torchvision are also included. Note that you still have to install detectron2 and register the datasets.

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

## Summary

To evaluate your model on the MESS benchmark, you can use the following steps:

- [ ] Prepare the datasets as described in [DATASETS.md](DATASETS.md)

- [ ] Register the datasets to Detectron2 by adding `import mess.datasets` to your evaluation code.

- [ ] Use the class names from `MetadataCatalog.get(dataset_name).stuff_classes` of each dataset.

- [ ] Use the `MESSSemSegEvaluator` as your evaluator class (optional).

For exemplary code changes, see [commit `1b5c5ee`](https://github.com/blumenstiel/CAT-Seg-MESS/commit/1b5c5ee103b60cc98af316f554c2a945a78856ca#diff-f4cc0633616b54356e2812aed5ce92d444e6d7c06673799b517fe6f74920a584) in <https://github.com/blumenstiel/CAT-Seg-MESS>.


## Citation

If you use the MESS benchmark, please cite our paper:

```
@article{MESSBenchmark2023,
  title={{What a MESS: Multi-Domain Evaluation of Zero-shot Semantic Segmentation}},
  author={Blumenstiel, Benedikt and Jakubik, Johannes and Kühne, Hilde and Vössing, Michael},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
