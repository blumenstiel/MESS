
# run python mess/prepare_datasets/prepare_cwfid.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image

train_ids = [2, 5, 6, 7, 8, 11, 12, 14, 16, 17, 18, 19, 20, 23, 24, 25, 27, 28, 31, 33, 34, 36, 37, 38, 40, 41, 42, 43,
             45, 46, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59]
test_ids = [1, 3, 4, 9, 10, 13, 15, 21, 22, 26, 28, 29, 30, 32, 35, 39, 44, 47, 48, 54, 60]


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    # Dataset page: https://github.com/cwfid/dataset.git
    print('Downloading dataset...')
    # Downloading dataset from git repo
    os.system('git clone https://github.com/cwfid/dataset.git')
    ds_path.mkdir(exist_ok=True, parents=True)
    os.system('mv dataset ' + str(ds_path))


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'cwfid'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['test']:
        # create directory
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(anno_dir, exist_ok=True)

        ids = test_ids if split == 'test' else train_ids
        for id in tqdm.tqdm(ids):
            # get mask path
            mask_path = ds_path / 'annotations' / f'{id:03}_annotation.png'
            # Open mask
            mask = np.array(Image.open(mask_path))

            # Edit annotations
            color_to_class = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0]}
            # Map RGB values to class index by converting to grayscale and applying a lookup table
            for class_idx, rgb in color_to_class.items():
                mask[(mask == rgb).all(axis=-1)] = class_idx
            mask = mask[:, :, 0]

            # Save mask
            Image.fromarray(mask).save(anno_dir / mask_path.name)

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
