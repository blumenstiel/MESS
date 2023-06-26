# Description: This script prepares all datasets
# Usage: python mess/prepare_all_datasets.py --dataset_dir datasets

import kaggle
import os
import argparse
from detectron2.data import DatasetCatalog
from prepare_datasets import (
    prepare_bdd100k,
    prepare_mhp_v1,
    prepare_foodseg,
    prepare_dark_zurich,
    prepare_atlantis,
    prepare_dram,
    prepare_isaid,
    prepare_isprs_potsdam,
    prepare_worldfloods,
    prepare_floodnet,
    prepare_uavid,
    prepare_kvasir_instrument,
    prepare_chase_db1,
    prepare_cryonuseg,
    prepare_paxray,
    prepare_pst900,
    prepare_corrosion_cs,
    prepare_deepcrack,
    prepare_zerowaste,
    prepare_suim,
    prepare_cub_200,
    prepare_cwfid,
)

if __name__ == '__main__':
    # parser to get dataset directory
    parser = argparse.ArgumentParser(description='Prepare datasets')
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='dataset directory')
    parser.add_argument('--stats', action='store_true', help='Only show dataset statistics')
    args = parser.parse_args()

    # set dataset directory and register datasets
    os.environ['DETECTRON2_DATASETS'] = args.dataset_dir
    os.makedirs(args.dataset_dir, exist_ok=True)
    import datasets

    # prepare datasets
    dataset_dict = {
        'dark_zurich_sem_seg_val': prepare_dark_zurich,
        'mhp_v1_sem_seg_test': prepare_mhp_v1,
        'foodseg103_sem_seg_test': prepare_foodseg,
        'atlantis_sem_seg_test': prepare_atlantis,
        'dram_sem_seg_test': prepare_dram,
        'isaid_sem_seg_val': prepare_isaid,
        'worldfloods_sem_seg_test_irrg': prepare_worldfloods,
        'kvasir_instrument_sem_seg_test': prepare_kvasir_instrument,
        'chase_db1_sem_seg_test': prepare_chase_db1,
        'cryonuseg_sem_seg_test': prepare_cryonuseg,
        'paxray_sem_seg_test_lungs': prepare_paxray,
        'pst900_sem_seg_test': prepare_pst900,
        'corrosion_cs_sem_seg_test': prepare_corrosion_cs,
        'deepcrack_sem_seg_test': prepare_deepcrack,
        'zerowaste_sem_seg_test': prepare_zerowaste,
        'suim_sem_seg_test': prepare_suim,
        'cub_200_sem_seg_test': prepare_cub_200,
        'cwfid_sem_seg_test': prepare_cwfid,

        ### Manual preparation ###
        # Place the manually downloaded zip files in the dataset directory or the root of the project.

        # Download 10k images and segmentation labels from https://bdd-data.berkeley.edu/ and place zip in datasets
        'bdd100k_sem_seg_val': prepare_bdd100k,
        # Download zip from https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx and place it in datasets
        'isprs_potsdam_sem_seg_test_irrg': prepare_isprs_potsdam,
        # Download folder from https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD to datasets
        'floodnet_sem_seg_test': prepare_floodnet,
        # Download zip from https://uavid.nl and place it in project root
        'uavid_sem_seg_val': prepare_uavid,
    }

    # print status of datasets
    print('Dataset: Status')
    for dataset_name in dataset_dict.keys():
        try:
            status = f'{len(DatasetCatalog.get(dataset_name))} images'
        except FileNotFoundError:
            status = 'Not found'
        except AssertionError:
            status = 'Not found'
        print(f'{dataset_name:50s} {status}')

    if args.stats:
        exit()

    for dataset_name, prepare_dataset in dataset_dict.items():
        # check if dataset is already prepared
        try:
            prepared = len(DatasetCatalog.get(dataset_name)) != 0
        except FileNotFoundError:
            prepared = False
        except AssertionError:
            prepared = False

        if prepared:
            print(f'\n{dataset_name} already prepared')
        else:
            # prepare dataset
            print(f'\nPreparing {dataset_name}')
            try:
                prepare_dataset.main()
            except Exception as e:
                print(f'Error while preparing {dataset_name}: \n{e}')

    # print status of datasets
    print('\nFinished preparing datasets')
    print('\nDataset: Status')
    for dataset_name in dataset_dict.keys():
        try:
            status = f'{len(DatasetCatalog.get(dataset_name))} images'
        except FileNotFoundError:
            status = 'Not found'
        except AssertionError:
            status = 'Not found'
        print(f'{dataset_name:50s} {status}')
