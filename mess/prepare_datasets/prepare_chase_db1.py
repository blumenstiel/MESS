
# run python mess/prepare_datasets/prepare_chased_b1.py

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
    # Downloading zip
    os.system('wget https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip')
    ds_path.mkdir(exist_ok=True, parents=True)
    os.system('unzip CHASEDB1.zip -d ' + str(ds_path))
    os.system('rm CHASEDB1.zip')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'CHASEDB1'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    TRAIN_LEN = 8
    for split in ['test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        for i, img_path in tqdm.tqdm(enumerate(sorted((ds_path).glob('*.jpg')))):
            if (i < TRAIN_LEN and split == 'test') or (i >= TRAIN_LEN and split == 'train'):
                continue

            # Move image
            img = Image.open(img_path)
            img = img.convert('RGB')
            img.save(img_dir / img_path.name)

            # Open mask
            id = img_path.stem
            mask = Image.open(ds_path / f'{id}_1stHO.png')
            # Edit annotations
            # Binary encoding: (0, 255) -> (0, 1)
            mask = np.array(mask).astype(np.uint8)
            # Save mask
            Image.fromarray(mask).save(anno_dir / f'{id}.png')

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
