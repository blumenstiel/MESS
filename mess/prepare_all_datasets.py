# Description: This script prepares all datasets
# Usage: python mess/prepare_all_datasets.py --dataset_dir datasets

import os
import argparse
try:
    from detectron2.data import DatasetCatalog
except:
    from mess.utils.catalog import DatasetCatalog

from mess.prepare_datasets import (
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


def check_datasets(dataset_list):
    expected_dataset_length = {
        'bdd100k_sem_seg_val': 1000,
        'dark_zurich_sem_seg_val': 50,
        'mhp_v1_sem_seg_test': 980,
        'foodseg103_sem_seg_test': 2125,
        'atlantis_sem_seg_test': 1295,
        'dram_sem_seg_test': 718,
        'isaid_sem_seg_val': 4055,
        'isprs_potsdam_sem_seg_test_irrg': 504,
        'worldfloods_sem_seg_test_irrg': 160,
        'floodnet_sem_seg_test': 5571,
        'uavid_sem_seg_val': 840,
        'kvasir_instrument_sem_seg_test': 118,
        'chase_db1_sem_seg_test': 20,
        'cryonuseg_sem_seg_test': 30,
        'paxray_sem_seg_test_lungs': 180,
        'corrosion_cs_sem_seg_test': 44,
        'deepcrack_sem_seg_test': 237,
        'zerowaste_sem_seg_test': 929,
        'pst900_sem_seg_test': 288,
        'suim_sem_seg_test': 110,
        'cub_200_sem_seg_test': 5794,
        'cwfid_sem_seg_test': 21,
    }

    # print status of datasets
    print('\nDataset status:')
    missing_datasets = []
    for dataset_name in dataset_list:
        try:
            length = len(DatasetCatalog.get(dataset_name))
            if length == 0:
                status = 'No images found'
            elif length == expected_dataset_length[dataset_name]:
                status = f'{length} images (OK)'
            else:
                status = f'{length} images ({expected_dataset_length[dataset_name]} images expected)'
                missing_datasets.append(dataset_name)
        except FileNotFoundError:
            status = 'Dataset not found'
            missing_datasets.append(dataset_name)
        except AssertionError:
            status = 'Dataset not found'
            missing_datasets.append(dataset_name)
        print(f'{dataset_name:50s} {status}')
    if len(missing_datasets) == 0:
        print('All datasets processed.')
    else:
        print(f'Found missing datasets or images in: {", ".join(missing_datasets)}.')


if __name__ == '__main__':
    # parser to get dataset directory
    parser = argparse.ArgumentParser(description='Prepare datasets')
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='dataset directory')
    parser.add_argument('--stats', action='store_true', help='Only show dataset statistics')
    args = parser.parse_args()

    # set dataset directory and register datasets
    os.environ['DETECTRON2_DATASETS'] = args.dataset_dir
    os.makedirs(args.dataset_dir, exist_ok=True)
    # Register datasets
    import mess.datasets

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

    check_datasets(dataset_dict.keys())

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
    check_datasets(dataset_dict.keys())
