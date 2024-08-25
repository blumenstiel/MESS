
# run python mess/prepare_datasets/prepare_cryonuseg.py

import os
import tqdm
import gdown
import numpy as np
from pathlib import Path
from PIL import Image


def check_dataset(dataset_dir, ds_path):
    """
    Check dataset and rename it
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)
    KAGGLE_DIR_NAME = 'archive'
    if os.path.exists(KAGGLE_DIR_NAME):
        # Move Kaggle dir to dataset directroy
        os.system(f'mv {KAGGLE_DIR_NAME} {dataset_dir}')

    assert os.path.exists(dataset_dir / KAGGLE_DIR_NAME), \
        ("Download dataset from Kaggle "
         "(https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images?resource=download) "
         f"and place the directroy {KAGGLE_DIR_NAME} into the project root or dataset directory.")

    # Rename dataset
    os.system(f'mv {dataset_dir / KAGGLE_DIR_NAME} {ds_path}')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'CryoNuSeg'
    if not ds_path.exists():
        check_dataset(dataset_dir, ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    # create directories
    img_dir = ds_path / 'images_detectron2'
    anno_dir = ds_path / 'annotations_detectron2'
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    for img_path in tqdm.tqdm((ds_path / 'tissue images').glob('*.tif')):
        id = img_path.stem
        # Move image
        img = Image.open(img_path)
        img = img.convert('RGB')
        img.save(img_dir / f'{id}.png', 'PNG')

        # Open mask
        mask = Image.open(ds_path / 'Annotator 1 (biologist second round of manual marks up)'
                          / 'Annotator 1 (biologist second round of manual marks up)' / 'mask binary' / f'{id}.png')
        # Edit annotations
        # Binary encoding: (0, 255) -> (0, 1)
        mask = np.uint8(np.array(mask) / 255)
        # Save mask
        Image.fromarray(mask).save(anno_dir / f'{id}.png')

    print(f'Saved images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
