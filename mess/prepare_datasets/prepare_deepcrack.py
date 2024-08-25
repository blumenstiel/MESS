
# run python mess/prepare_datasets/prepare_deepcrack.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading git repo with zip
    os.system('git clone https://github.com/yhlleo/DeepCrack.git')
    ds_path.mkdir(exist_ok=True, parents=True)
    os.system('unzip DeepCrack/dataset/DeepCrack.zip -d ' + str(ds_path))
    os.system('rm -R DeepCrack')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'DeepCrack'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['test']:
        # create directory
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(anno_dir, exist_ok=True)

        for mask_path in tqdm.tqdm((ds_path / f'{split}_lab').glob('*.png')):
            # Open mask
            mask = Image.open(mask_path)
            # Edit annotations
            # Binary encoding: (0, 255) -> (0, 1)
            mask = np.uint8(np.array(mask) / 255)
            # Save mask
            Image.fromarray(mask).save(anno_dir / mask_path.name)

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
