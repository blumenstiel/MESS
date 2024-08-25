# Datasets for Multi-domain Evaluation of Semantic Segmentation (MESS) 

## Download datasets

Most datasets can be downloaded automatically by a preparation script. 

Some manual downloads are required:
- BDD100K: https://bdd-data.berkeley.edu/ (`10K Images` and `Segmentation`)
- FloodNet: https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD (Download Directory or test folder)
- ISPRS Potsdam: https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx (Potsdam)
- UAVid: https://uavid.nl (Download -> Semantic Labelling with Images Only)
- CryoNuSeg: https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images (Download from Kaggle; Directory name 'archive')

You can place the downloaded files in the project root or unzip in your data directory.

## Install environment

The processing of the datasets require certain packages. You can install all pypi packages with `pip install mess-benchmark`.
Note, that the DRAM dataset is compressed in a rar file. You may need to install unrar to extract it: `sudo apt install unrar`.

If you plan to use `detectron2` for your modeling, we provide a script for setting up the environment in [setup_env.sh](setup_env.sh).
See https://detectron2.readthedocs.io/en/latest/tutorials/install.html for further install instructions. `detectron2` is not required for loading the datasets.

## Dataset download and preparation

You can download and prepare all datasets by running the following script. Note that the datasets are provided by external parties and are not associated with this repository. Please consider the terms and conditions of each dataset, which are linked below.

```bash
python -m mess.prepare_all_datasets --dataset_dir datasets

# your can check the preparation with
python -m mess.prepare_all_datasets --dataset_dir datasets --stats
```

Four datasets require a manual download that are listed above.
If the automatic downloads do not work, please consider the descriptions below.

If you are using another dataset directory than `datasets`, you have to export it as the `DETECTRON2_DATASETS` environment variable before evaluating the models (see `eval.sh`). E.g, `export DETECTRON2_DATASETS=../mess_datasets` when evaluating multiple models.

## Dataset overview
We provide an overview of all datasets in the following. Please note the different licenses of the datasets. 
If you use the datasets, please cite the corresponding papers. 
We provide the BibTeX for all datasets in [datasets.bib](datasets.bib).

The scripts currently only support the download and preprocessing for the val/test splits.

Datasets:
1. [BDD100K](#bdd100k)
2. [Dark Zurich](#dark-zurich)
3. [MHP v1](#mhp-v1)
4. [FoodSeg103](#foodseg103) 
5. [ATLANTIS](#atlantis)
6. [DRAM](#dram)
7. [iSAID](#isaid)
8. [ISPRS Potsdam](#isprs-potsdam)
9. [WorldFloods](#worldfloods)
10. [FloodNet](#floodnet)
11. [UAVid](#uavid)
12. [Kvasir-Instrument](#kvasir-instrument)
13. [CryoNuSeg](#cryonuseg)
14. [CHASE DB1](#chase-bd1)
15. [PAXRay-4](#paxray-4)
16. [Corrosion CS](#corrosion-cs)
17. [DeepCrack](#deepcrack)
18. [PST900](#pst900)
19. [ZeroWaste-f](#zerowaste-f)
20. [SUIM](#suim)
21. [CUB-200](#cub-200)
22. [CWFID](#cwfid)



### Custom datasets

We provide two templates to provide guidance for adding custom datasets: `mess.prepare_datasets.prepare_TEMPLATE` and `mess.datasets/register_TEMPLATE`.

### BDD100K

Registered name: `bdd100k_sem_seg_val`

Dataset page: https://bdd-data.berkeley.edu

Paper: Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., ... & Darrell, T. (2020). Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 2636-2645).

Licence: https://doc.bdd100k.com/license.html

Non-commercial use only. Commercial use only for BDD members.

Download the dataset from https://bdd-data.berkeley.edu/ (bdd100k_images_10k.zip and bdd100k_sem_seg_labels_trainval.zip) and place the zip files in the project root or datasets folder. Prepare the dataset by running:
Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_bdd100k
```

### Dark Zurich

Registered name: `dark_zurich_sem_seg_val`

Dataset page: https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/

Paper: Sakaridis, C., Dai, D., & Gool, L. V. (2019). Guided curriculum model adaptation and uncertainty-aware evaluation for semantic nighttime image segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 7374-7383).

Licence: custom (in dataset download)

Citation required. Non-commercial use only.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_dark_zurich
```
If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Download: https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip

### MHP v1

Registered name: `mhp_v1_sem_seg_test`

Dataset page: https://github.com/ZhaoJ9014/Multi-Human-Parsing

Paper: Li, J., Zhao, J., Wei, Y., Lang, C., Li, Y., Sim, T., ... & Feng, J. (2017). Multiple-human parsing in the wild. arXiv preprint arXiv:1705.07206.

Licence: https://lv-m hp.github.io/

Non-commercial use only.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_mhp_v1
```
If the download does not work, download the dataset manually from Google Drive and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Google Drive: https://drive.google.com/uc?export=download&confirm=pbef&id=1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5

### FoodSeg103

Registered name: `foodseg103_sem_seg_test`

Dataset page: https://xiongweiwu.github.io/foodseg103.html

Paper: Wu, X., Fu, X., Liu, Y., Lim, E. P., Hoi, S. C., & Sun, Q. (2021, October). A large-scale benchmark for food image segmentation. In Proceedings of the 29th ACM International Conference on Multimedia (pp. 506-515).

Licence: [Apache 2.0](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1/blob/main/LICENSE)

Copyright notice required.

Download the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_foodseg
```
If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`).

Download: https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip

### ATLANTIS

Registered name: `atlantis_sem_seg_test`

Dataset page: https://github.com/smhassanerfani/atlantis

Paper: Erfani, S. M. H., Wu, Z., Wu, X., Wang, S., & Goharian, E. (2022). ATLANTIS: A benchmark for semantic segmentation of waterbody images. Environmental Modelling & Software, 149, 105333.

Licence: https://github.com/smhassanerfani/atlantis

Citation requested. Images are under the Flickr terms of use.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_atlantis
```
If the download does not work, download the git repo and place the subfolder `atlanis/atlanis` as folder `atlantis` in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

### DRAM

Registered name: `dram_sem_seg_test`

Dataset page: https://faculty.runi.ac.il/arik/site/artseg/Dram-Dataset.html

Paper: Cohen, N., Newman, Y., & Shamir, A. (2022, May). Semantic Segmentation in Art Paintings. In Computer Graphics Forum (Vol. 41, No. 2, pp. 261-275).

Licence: https://faculty.runi.ac.il/arik/site/artseg/Dram-Dataset.html

Citation required. Non-commercial use only.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_dram
```
The DRAM dataset is compressed in a rar file. You may need to install unrar to extract it: `sudo apt install unrar`.

DRAM does not include a labeled training dataset but requires domain transfer from Pascal VOC.

If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Download: https://faculty.runi.ac.il/arik/site/artseg/DRAM_processed.zip

### iSAID

Registered name: `isaid_sem_seg_val`

Dataset page: https://captain-whu.github.io/iSAID/dataset.html

Paper: Waqas Zamir, S., Arora, A., Gupta, A., Khan, S., Sun, G., Shahbaz Khan, F., ... & Bai, X. (2019). isaid: A large-scale dataset for instance segmentation in aerial images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 28-37).

Licence: https://captain-whu.github.io/iSAID/dataset.html

Citation requested. Non-commercial use only.

Images are under the "Google Earth" terms of use.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_isaid
```
The original dataset consists of very large tiles. We are using non-overlapping 1024x1024 patches with zero-padding, which is ignored during evaluation.

If the download does not work, download the dataset manually from Google Drive and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Google Drive images: https://drive.google.com/drive/folders/1RV7Z5MM4nJJJPUs6m9wsxDOJxX6HmQqZ?usp=share_link4

Google Drive masks: https://drive.google.com/drive/folders/1jlVr4ClmeBA01IQYx7Aq3Scx2YS1Bmpb

### ISPRS Potsdam

Registered name: `isprs_potsdam_sem_seg_test_irrg` (IRRG, used in MESS) and `isprs_potsdam_sem_seg_test_rgb` (RGB)

Dataset page: https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx

Dataset provider: BSF Swissphoto (2012). Isprs potsdam dataset within the isprs test project on urban classification, 3d building reconstruction and semantic labeling. at https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx

Licence: undefined

Upon personal request to the challenge organizers: Citation required. Notice of BSF Swissphoto as dataset provider.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_isprs_potsdam
```
The script creates an IRRG and RGB colormap. The main evaluation uses the IRRG images.

The original dataset has tiles of 6000x6000. We are using non-overlapping 1024x1024 patches with zero-padding, which is ignored during evaluation. 

If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Unzip the folders for RGB, IRRG and Labels_all. Then run the script again.

Download: https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx

### WorldFloods

Registered name: `worldfloods_sem_seg_test_irrg` (IRRG, used in MESS) and `worldfloods_sem_seg_test_rgb` (RGB)

Dataset page: https://spaceml-org.github.io/ml4floods/content/worldfloods_dataset.html

Dataset download: https://gigatron.uv.es/owncloud/index.php/s/JhNwsrrwDt80Vqc

Paper: Mateo-Garcia, G., Veitch-m ichaelis, J., Smith, L., Oprea, S. V., Schumann, G., Gal, Y., ... & Backes, D. (2021). Towards global flood mapping onboard low cost satellites with machine learning. Scientific reports, 11(1), 1-12.

Licence: [CC NC 4.0] (https://gigatron.uv.es/owncloud/index.php/s/JhNwsrrwDt80Vqc#editor)

Citation (Attribution) required. Non-commercial use only.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_worldfloods
```
The script creases an IRRG and RGB colormap. The main evaluation uses the IRRG images.

The original dataset consists of very large tiles. We are using non-overlapping 1024x1024 patches with zero-padding, which is ignored during evaluation.

We do not distinguish between flood and permanent water as this task is very difficult without additional data.

If the download does not work, download the test set manually from Google Drive and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Google Drive: https://drive.google.com/drive/folders/1Bp1FXppikOpQrgth2lu5WjpYX7Lb2qOW?usp=share_link

### FloodNet

Registered name: `floodnet_sem_seg_test`

Dataset page: https://github.com/BinaLab/FloodNet-Supervised_v1.0

Paper: Rahnemoonfar, M., Chowdhury, T., Sarkar, A., Varshney, D., Yari, M., & Murphy, R. R. (2021). Floodnet: A high resolution aerial imagery dataset for post flood scene understanding. IEEE Access, 9, 89644-89654.

Licence: https://cdla.dev/permissive-1-0/

Citation and link required.

Download the dataset from Google Drive and place it in the dataset folder as `FloodNet-Supervised_v1.0`. No further preprocessing is needed. You can check if the dataset is working by running:
```sh
python -m mess.prepare_datasets.prepare_floodnet
```
The original dataset consists of very large images. We are using non-overlapping 1024x1024 patches with zero-padding, which is ignored during evaluation.


Google Drive: https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD

### UAVid

Registered name: `uavid_sem_seg_val`

Dataset page: https://uavid.nl

Paper: Lyu, Y., Vosselman, G., Xia, G. S., Yilmaz, A., & Yang, M. Y. (2020). UAVid: A semantic segmentation dataset for UAV imagery. ISPRS journal of photogrammetry and remote sensing, 165, 108-119.

Licence: [CC BY-NC-SA 4.0](https://uavid.nl/)

Citation (Attribution) required. Non-commercial use only.

Please download the dataset from the website and place it in the project or dataset folder. Prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_uavid
```
The original dataset consists of very large tiles. We are using non-overlapping 1024x1024 patches with zero-padding, which is ignored during evaluation.

### Kvasir-Instrument

Registered name: `kvasir_instrument_sem_seg_test`

Dataset page: https://datasets.simula.no/kvasir-instrument/

Paper: Jha, D., Ali, S., Emanuelsen, K., Hicks, S. A., Thambawita, V., Garcia-Ceja, E., ... & Halvorsen, P. (2021). Kvasir-instrument: Diagnostic and therapeutic tool segmentation dataset in gastrointestinal endoscopy. In MultiMedia Modeling: 27th International Conference, MMM 2021, Prague, Czech Republic, June 22–24, 2021, Proceedings, Part II 27 (pp. 218-229). Springer International Publishing.

Licence: https://datasets.simula.no/kvasir-instrument/

Citation required. Non-commercial use only.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_kvasir_instrument
```
If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Also untar the image folders. Then run the script again.

Download: https://datasets.simula.no/downloads/kvasir-instrument.zip

### CHASE BD1

Registered name: `chase_db1_sem_seg_test`

Dataset page: https://blogs.kingston.ac.uk/retinal/chasedb1/

Paper: Fraz, M. M., Remagnino, P., Hoppe, A., Uyyanonvara, B., Rudnicka, A. R., Owen, C. G., & Barman, S. A. (2012). An ensemble classification-based approach applied to retinal blood vessel segmentation. IEEE Transactions on Biomedical Engineering, 59(9), 2538-2548.

Licence: CC BY 4.0 (https://researchdata.kingston.ac.uk/96/)

Citation (Attribution) required.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_chase_db1
```

If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Download: https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip

### CryoNuSeg

Registered name: `cryonuseg_sem_seg_test`

Dataset page: https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images

Paper: Mahbod, A., Schaefer, G., Bancher, B., Löw, C., Dorffner, G., Ecker, R., & Ellinger, I. (2021). CryoNuSeg: A dataset for nuclei instance segmentation of cryosectioned H&E-stained histological images. Computers in biology and medicine, 132, 104349.

Licence: CC BY-NC-SA 4.0 (https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images)

Citation (Attribution) required. Non-commercial use only.

Download the dataset manually from [Kaggle](https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images), place the directory into the project root and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_cryonuseg
```
CryoNuSeg only includes a test set. Other datasets can be used for training.

### PAXRay-4

Registered name: `paxray_sem_seg_test_lungs paxray_sem_seg_test_bones paxray_sem_seg_test_mediastinum paxray_sem_seg_test_diaphragm` (Multi-label segmentation)

Dataset page: https://constantinseibold.github.io/paxray/

Paper: Seibold, C., Reiß, S., Sarfraz, S., Fink, M. A., Mayer, V., Sellner, J., ... & Stiefelhagen, R. (2022). Detailed Annotations of Chest X-Rays via CT Projection for Report Understanding. arXiv preprint arXiv:2210.03416.

Licence: https://constantinseibold.github.io/paxray/

Citation requested.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_paxray
```
We are only evaluating on the 4 superclasses (lungs, mediastinum, bones, diaphragm). Because of overlapping masks, each class is a binary segmentation tasks. We average the results of each class.

If the download does not work, download the dataset manually from Google Drive and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Google Drive: https://drive.google.com/file/d/19HPPhKf9TDv4sO3UV-nI3Jhi4nCv_Zyc/view?usp=share_link

### Corrosion CS

Registered name: `corrosion_cs_sem_seg_test`

Dataset page: https://figshare.com/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663

Paper: Bianchi, E., & Hebdon, M. (2021). Corrosion condition state semantic segmentation dataset. University Libraries, Virginia Tech: Blacksburg, VA, USA.

Licence: CCO (https://figshare.com/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663)

Citation requested.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_corrosion_cs
```
If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Download: https://figshare.com/ndownloader/files/31729733

### DeepCrack

Registered name: `deepcrack_sem_seg_test`

Dataset page: https://github.com/yhlleo/DeepCrack/tree/master

Paper: Liu, Y., Yao, J., Lu, X., Xie, R., & Li, L. (2019). DeepCrack: A deep hierarchical feature learning architecture for crack segmentation. Neurocomputing, 338, 139-153.

Licence: https://github.com/yhlleo/DeepCrack/tree/master

Citation requested.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_deepcrack
```
If the download does not work, download the git repo manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

### PST900

Registered name: `pst900_sem_seg_test` (thermal, used in MESS), `pst900_sem_seg_test_rgb` (RGB), `pst900_sem_seg_test_pseudo` (Thermal pseudo color map)

Dataset page: https://github.com/ShreyasSkandanS/pst900_thermal_rgb

Paper: Shivakumar, S. S., Rodrigues, N., Zhou, A., Miller, I. D., Kumar, V., & Taylor, C. J. (2020, May). Pst900: Rgb-thermal calibration, dataset and segmentation network. In 2020 IEEE international conference on robotics and automation (ICRA) (pp. 9441-9447). IEEE.

Licence: [GPL-3.0](https://github.com/ShreyasSkandanS/pst900_thermal_rgb/blob/master/LICENSE)

Citation required. 

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_pst900
```
We are using the thermal images as input (grayscale).

If the download does not work, download the dataset manually from Google Drive and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Run the script again to generate thermal images with a pseudo colorscale.

Google Drive: https://drive.google.com/open?id=1hZeM-m vdUC_Btyok7mdF00RV-InbAadm

### ZeroWaste-f

Registered name: `zerowaste_sem_seg_test`

Dataset page: http://ai.bu.edu/zerowaste/

Paper: Bashkirova, D., Abdelfattah, M., Zhu, Z., Akl, J., Alladkani, F., Hu, P., ... & Saenko, K. (2022). Zerowaste dataset: Towards deformable object segmentation in cluttered scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 21147-21157).

Licence: [CC-BY-NC 4.0](http://ai.bu.edu/zerowaste/)

Citation (Attribution) required. Non-commercial use only.

Download the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_zerowaste
```
If the download does not work, download the dataset manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`).

Download: https://zenodo.org/record/6412647/files/zerowaste-f-final.zip

### SUIM

Registered name: `suim_sem_seg_test`

Dataset page: https://irvlab.cs.umn.edu/resources/suim-dataset

Paper: Islam, M. J., Edge, C., Xiao, Y., Luo, P., Mehtaz, M., Morse, C., ... & Sattar, J. (2020, October). Semantic segmentation of underwater imagery: Dataset and benchmark. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 1769-1776). IEEE.

Licence: Unknown or [MIT](https://github.com/xahidbuffon/SUIM/blob/master/LICENSE)

It is unclear whether the dataset is licensed under MIT or if the license cover only the provided code. We requested the dataset provider for clarification. 

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_suim
```
If the download does not work, download the dataset manually from Google Drive and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Google Drive: https://drive.google.com/drive/folders/10KMK0rNB43V2g30NcA1RYipL535DuZ-h

### CUB-200

Registered name: `cub_200_sem_seg_test`

Dataset page: https://www.vision.caltech.edu/datasets/cub_200_2011/

Paper: Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The caltech-ucsd birds-200-2011 dataset. 

Licence: https://www.vision.caltech.edu/datasets/cub_200_2011/

Citation required. Non-commercial use only.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_cub_200
```
If the download does not work, download the images and annotations manually and place the unzipped dataset in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

Downloads:

Images: https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

Annotations: https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz

### CWFID

Registered name: `cwfid_sem_seg_test`

Dataset page: https://github.com/cwfid/dataset

Paper: Haug, S., & Ostermann, J. (2015). A crop/weed field image dataset for the evaluation of computer vision based precision agriculture tasks. In Computer Vision-ECCV 2014 Workshops: Zurich, Switzerland, September 6-7 and 12, 2014, Proceedings, Part IV 13 (pp. 105-116). Springer International Publishing.

Licence: https://github.com/cwfid/dataset

Citiation requested. Non-commercial use only.

Download and prepare the dataset by running:
```sh
python -m mess.prepare_datasets.prepare_cwfid
```
If the download does not work, download the git repo manually and place the dataset folder as `cwfid` in your Detectron2 dataset folder (default: `datasets/`). Then run the script again.

### BibTeX

We provide all BibTeX entries for the dataset papers in the [datasets.bib](datasets.bib) file. The BibTeX entries are named after the dataset with the pattern "Dataset<Name>", while " " and "-" are removed from the name. 