
# run python mess/prepare_datasets/prepare_<DATASET NAME>.py

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
    # TODO: Add an automated script if possible, otherwise remove code
    ds_path.mkdir(exist_ok=True, parents=True)
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://link/to/DATASET.zip')
    os.system('unzip <DATASET>.zip -d ' + str(ds_path))
    os.system('rm <DATASET>.zip')

    # Downloading tar.gz
    os.system('wget https://link/to/DATASET.tar.gz')
    os.system(f'tar -xvzf <DATASET>.tar.gz -C {ds_path}')
    os.system(f'rm <DATASET>.tar.gz')

    # Download from Google Drive
    gdown.download_folder(id='<FOLDER ID>', output=str(ds_path))
    gdown.download(id='<FILE ID>')
    # When access is denied
    gdown.download(f"https://drive.google.com/uc?export=download&confirm=pbef&id=<FILE ID>")
    os.system('unzip <DATASET>.zip -d ' + str(ds_path))
    os.system('rm <DATASET>.zip')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / '<DATASET>'
    # TODO: Remove if an automated download is not possible
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['test']:
        # TODO: Change if other directories are required
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        # TODO: Change image directory to the current split. Change the image extension if necessary
        for img_path in tqdm.tqdm((ds_path / '<OLD IMAGE DIR>' / split).glob('*.jpg')):
            # Copy image
            img = Image.open(img_path)
            img = img.convert('RGB')
            img.save(img_dir / img_path.name)

            # Open mask
            # TODO: Load the mask by using the image id or changing the image path
            id = img_path.stem
            mask = np.array(Image.open(ds_path / '<OLD IMAGE DIR>' / split / f'{id}.png'))
            # TODO: Alternative, if the mask is in the same directory as the image
            mask = np.array(Image.open(str(img_path).replace('.jpg', '.png')))

            # Edit annotations
            # TODO: Map the current annotations to the right format (classes 0, 1, ...; ignore value 255)
            # TODO: When more than 255 classes are used, use tif mask files with ignore value 65536
            # Binary encoding: (0, 255) -> (0, 1)
            mask = np.uint8(mask / 255)

            # Replace grey values with class index
            color_to_class = {11: 0, 22: 1}
            mask = np.vectorize(color_to_class.get)(mask).astype(np.uint8)

            # Map RGB values to class index by converting to grayscale and applying a lookup table
            color_to_class = {0: [255, 255, 255], 1: [255, 0, 0], }
            for class_idx, rgb in color_to_class.items():
                mask[(mask == rgb).all(axis=-1)] = class_idx
            mask = mask[:, :, 0]

            # Save mask
            Image.fromarray(mask).save(anno_dir / f'{id}.png')

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
