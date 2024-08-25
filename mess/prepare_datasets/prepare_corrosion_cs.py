
# run python mess/prepare_datasets/prepare_corrosion_cs.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image


# Classes:
# 0: background
# 1: Fair
# 2: Poor
# 3: Severe

def download_dataset(dataset_dir):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading zip
    #
    os.system('wget https://figshare.com/ndownloader/files/31729733')
    dataset_dir.mkdir(exist_ok=True, parents=True)
    os.system('unzip 31729733 -d ' + str(dataset_dir))
    os.system('rm 31729733')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'Corrosion Condition State Classification'
    if not ds_path.exists():
        download_dataset(dataset_dir)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    # Edit test masks
    for mask_path in tqdm.tqdm(sorted(ds_path.glob('original/Test/masks/*.png'))):
        # Open mask
        mask = np.array(Image.open(mask_path))
        # 'Portable network graphics' format, so no further processing needed
        # Save mask
        Image.fromarray(mask).save(mask_path)

    print(f'Saved images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
