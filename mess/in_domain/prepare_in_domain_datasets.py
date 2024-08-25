# Description: This script prepares all datasets
# Usage: python in_domain/prepare_in_domain_datasets.py --dataset_dir datasets

import os
import argparse

try:
    from detectron2.data import DatasetCatalog
except:
    from mess.utils.catalog import DatasetCatalog

from mess.in_domain.prepare_datasets import (
    prepare_ade20k_sem_seg,
    prepare_ade20k_full_sem_seg,
    prepare_pascal_context_59,
    prepare_pascal_context_459,
    prepare_pascal_voc,
)

if __name__ == '__main__':
    # parser to get dataset directory
    parser = argparse.ArgumentParser(description='Prepare in-domain datasets')
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='dataset directory')
    parser.add_argument('--stats', action='store_true', help='Only show dataset statistics')
    args = parser.parse_args()

    # set dataset directory
    os.environ['DETECTRON2_DATASETS'] = args.dataset_dir
    os.makedirs(args.dataset_dir, exist_ok=True)

    import mess.in_domain.datasets

    # prepare datasets
    dataset_dict = {
        'ade20k_sem_seg_val': prepare_ade20k_sem_seg,
        'pascal_context_59_sem_seg_val': prepare_pascal_context_59,
        'pascal_context_459_sem_seg_val': prepare_pascal_context_459,
        'voc_2012_sem_seg_val_bg': prepare_pascal_voc,

        # # # Manual preparation # # #
        # Place the manually downloaded zip files in the dataset directory or the root of the project.

        # Download ADE20K-847 from https://groups.csail.mit.edu/vision/datasets/ADE20K/request_data/.
        'ade20k_full_sem_seg_val': prepare_ade20k_full_sem_seg,
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
    print('\nFinished preparing in-domain datasets')
    print('\nDataset: Status')
    for dataset_name in dataset_dict.keys():
        try:
            status = f'{len(DatasetCatalog.get(dataset_name))} images'
        except FileNotFoundError:
            status = 'Not found'
        except AssertionError:
            status = 'Not found'
        print(f'{dataset_name:50s} {status}')
