
# run python mess/prepare_datasets/prepare_pst900.py

import os
import tqdm
import gdown
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Using 'inferno' color map for thermal images
inferno_colormap = plt.get_cmap('inferno')


def download_dataset(dataset_dir):
    """
    Downloads the dataset
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)
    print('Downloading dataset...')
    # Download from Google Drive
    # link: https://drive.google.com/open?id=1hZeM-MvdUC_Btyok7mdF00RV-InbAadm
    gdown.download("https://drive.google.com/uc?export=download&confirm=pbef&id=1hZeM-MvdUC_Btyok7mdF00RV-InbAadm")
    os.system('unzip PST900_RGBT_Dataset.zip -d ' + str(dataset_dir))
    os.system('rm PST900_RGBT_Dataset.zip')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'PST900_RGBT_Dataset'
    if not ds_path.exists():
        download_dataset(dataset_dir)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    # create directories
    thermal_dir = ds_path / 'test' / 'thermal_pseudo'
    os.makedirs(thermal_dir, exist_ok=True)

    for img_path in tqdm.tqdm((ds_path / 'test' / 'thermal').glob('*.png')):
        # Open image
        img = Image.open(img_path)
        # Change thermal gray scale to pseudo color
        img = inferno_colormap(np.array(img)) * 255
        img = img.astype(np.uint8)[:, :, :3]
        # Save thermal pseudo color image
        Image.fromarray(img).save(thermal_dir / img_path.name)

    print(f'Saved images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
