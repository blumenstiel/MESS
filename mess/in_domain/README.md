# In-domain Evaluation for Zero-shot Semantic Segmentation

[[Website](https://blumenstiel.github.io/mess-benchmark/)] [[arXiv](https://arxiv.org/abs/2306.15521)] [[GitHub](https://github.com/blumenstiel/MESS)]

This directory contains the code for the preparation and registration of five in-domain datasets that are typically used in zero-shot semantic segmentation. We refer to the [supplementary material](https://arxiv.org/abs/2306.15521) for further details of the datasets and the results from the evaluated models. The preparation code is similar to the scripts from [CAT-Seg](https://github.com/KU-CVLAB/CAT-Seg). 

For more information about the usage of this directory, see [GettingStarted.md](../../GettingStarted.md).

## Prepare the datasets

Download the data of ADE20k-847 from https://groups.csail.mit.edu/vision/datasets/ADE20K/request_data/. 

We added scripts to download four datasets automatically: Pascal VOC 2012, Pascal Context-59, Cityscapes, Pascal Context-459, and ADE20K-150. Note that the datasets are provided by external parties and are not associated with this repository. Please consider the terms and conditions of each dataset, which are linked below.

See [DATASETS.md](DATASETS.md) for install instructions. Prepare all five in-domain datasets with the following script: 

```bash
python -m mess.in_domain.in_domain/prepare_in_domain_datasets --dataset_dir datasets

# your can check the preparation with
python -m mess.in_domain.in_domain/prepare_in_domain_datasets --dataset_dir datasets --stats
```

If the automatic downloads do not work, please consider the descriptions below.

If you are using another dataset directory than `datasets`, you have to export it as the `DETECTRON2_DATASETS` environment variable before evaluating the models (see `mess.in_domain.in_domain/eval.sh`). E.g, `export DETECTRON2_DATASETS=../in_domain_datasets` when evaluating multiple models.

## In-domain dataset overview

We provide an overview of all datasets in the following. Please note the different licenses of the datasets.

Datasets:
1. [ADE20K-150](#ade20k-150)
2. [ADE20K-847](#ade20k-847)
3. [Pascal Context-59](#pascal-context-59)
4. [Pascal Context-459](#pascal-context-459) 
5. [Pascal VOC](#pascal-voc)

### ADE20K-150

Registered name: `ade20k_sem_seg_val`

Dataset page: https://groups.csail.mit.edu/vision/datasets/ADE20K/

Paper: Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., & Torralba, A. (2017). Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 633-641).

Licence: https://groups.csail.mit.edu/vision/datasets/ADE20K/terms/

Citation requested. Non-commercial use only.

Download and prepare the dataset by running:
```sh
python -m in_domain/prepare_datasets.prepare_ade20k_sem_seg
```
If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Download: http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip


### ADE20K-847

Registered name: `ade20k_full_sem_seg_val`

Dataset page: https://groups.csail.mit.edu/vision/datasets/ADE20K/

Paper: Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., & Torralba, A. (2017). Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 633-641).

Licence: https://groups.csail.mit.edu/vision/datasets/ADE20K/terms/

Citation requested. Non-commercial use only. 

Download the dataset from https://groups.csail.mit.edu/vision/datasets/ADE20K/request_data/ and place the zip file in the project root or datasets folder. Prepare the dataset by running:
```sh
python -m in_domain/prepare_datasets.prepare_ade20k_full_sem_seg
```

### Pascal Context-59

Registered name: `pascal_context_59_sem_seg_val`

Dataset page: https://cs.stanford.edu/~roozbeh/pascal-context/

Paper: Mottaghi, R., Chen, X., Liu, X., Cho, N. G., Lee, S. W., Fidler, S., ... & Yuille, A. (2014). The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 891-898).

Licence: "flickr" terms of use (http://host.robots.ox.ac.uk/pascal/VOC/voc2010/#rights)

Download and prepare the dataset by running:
```sh
python -m mess.in_domain.prepare_datasets.prepare_pascal_context_59
```
If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Downloading images: http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar

Downloading val split: https://drive.google.com/file/d/1BCbiOKtLvozjVnlTJX51koIveUZHCcUh/view

Download 59 labels: https://codalabuser.blob.core.windows.net/public/trainval_merged.json

### Pascal Context-459

Registered name: `pascal_context_459_sem_seg_val`

Dataset page: https://cs.stanford.edu/~roozbeh/pascal-context/

Paper: Mottaghi, R., Chen, X., Liu, X., Cho, N. G., Lee, S. W., Fidler, S., ... & Yuille, A. (2014). The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 891-898).

Licence: "flickr" terms of use (http://host.robots.ox.ac.uk/pascal/VOC/voc2010/#rights)

Download and prepare the dataset by running:
```sh
python -m mess.in_domain.prepare_datasets.prepare_pascal_context_459
```
If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again. The images and val split is similar to Pascal Context-59. 

Downloading images: http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar

Downloading val split: https://drive.google.com/file/d/1BCbiOKtLvozjVnlTJX51koIveUZHCcUh/view

Download 459 labels: https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz

### Pascal VOC

Registered name: `voc_2012_sem_seg_val_bg`

Dataset page: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

Paper: Everingham, M., Van Gool, L., Williams, C. K., Winn, J., & Zisserman, A. (2010). The pascal visual object classes (voc) challenge. International journal of computer vision, 88, 303-338.

Licence: "flickr" terms of use (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#rights)

Download and prepare the dataset by running:
```sh
python -m mess.in_domain.prepare_datasets.prepare_pascal_voc
```
If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Download images: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

Download masks: https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip 
