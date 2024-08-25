
# run python mess/prepare_datasets/prepare_bdd100k.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image


def download_dataset(dataset_dir):
    """
    Downloads the dataset
    """
    print('Unzip dataset...')
    zip_dir = Path('') if os.path.isfile('bdd100k_images_10k.zip') else Path(dataset_dir)
    assert os.path.isfile(zip_dir / 'bdd100k_images_10k.zip'), \
        'bdd100k_images_10k.zip not found, ' \
        'please download 10K images and segmentation from https://bdd-data.berkeley.edu/ and place zip in the dataset dir'
    assert os.path.isfile(zip_dir / 'bdd100k_sem_seg_labels_trainval.zip'), \
        'bdd100k_sem_seg_labels_trainval.zip not found, ' \
        'please download 10K images and segmentation from https://bdd-data.berkeley.edu/ and place zip in the dataset dir'

    os.system(f'unzip {zip_dir / "bdd100k_images_10k.zip"} -d {dataset_dir}')
    os.system(f'unzip {zip_dir / "bdd100k_sem_seg_labels_trainval.zip"} -d {dataset_dir}')
    dataset_dir.mkdir(exist_ok=True, parents=True)
    os.system(f'rm {zip_dir / "bdd100k_images_10k.zip"}')
    os.system(f'rm {zip_dir / "bdd100k_sem_seg_labels_trainval.zip"}')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'bdd100k'
    if not ds_path.exists():
        download_dataset(dataset_dir)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['val']:
        # create directories
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(anno_dir, exist_ok=True)

        for mask_path in tqdm.tqdm(list((ds_path / 'labels/sem_seg/colormaps' / split).glob('*.png'))):
            # Open mask
            mask = Image.open(mask_path)
            # Mapping color map to class id
            mask = Image.fromarray(np.array(mask))
            # Save mask
            mask.save(anno_dir / mask_path.name)

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
