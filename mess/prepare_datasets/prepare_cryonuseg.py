
# run python mess/prepare_datasets/prepare_cryonuseg.py

import os
import tqdm
import gdown
import numpy as np
from pathlib import Path
from PIL import Image


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Download from Google Drive
    # Folder: https://drive.google.com/drive/folders/1dgtO_mCcR4UNXw_4zK32NlakbAvnySck
    gdown.download("https://drive.google.com/uc?export=download&confirm=pbef&id=1Or8qSpwLx77ZcWFqOKCKd3upwTUvb0U6")
    gdown.download("https://drive.google.com/uc?export=download&confirm=pbef&id=1WHork0VjF1PTye1xvCTtPtly62uHF72J")
    os.makedirs(ds_path, exist_ok=True)
    os.system('unzip Final.zip -d ' + str(ds_path))
    os.system('unzip masks.zip -d ' + str(ds_path))
    os.system('rm Final.zip')
    os.system('rm masks.zip')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'CryoNuSeg'
    if not ds_path.exists():
        download_dataset(ds_path)

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
        mask = Image.open(ds_path / 'mask binary' / f'{id}.png')
        # Edit annotations
        # Binary encoding: (0, 255) -> (0, 1)
        mask = np.uint8(np.array(mask) / 255)
        # Save mask
        Image.fromarray(mask).save(anno_dir / f'{id}.png')

    print(f'Saved images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
