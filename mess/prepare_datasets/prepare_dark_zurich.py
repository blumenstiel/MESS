
# run python mess/prepare_datasets/prepare_dark_zurich.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image

DARK_ZURICH_LABELS = (0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
labels_to_id = {label: i for i, label in enumerate(DARK_ZURICH_LABELS)}
labels_to_id[255] = 255


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    # Dataset page: https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip')
    ds_path.mkdir(exist_ok=True, parents=True)
    os.system('unzip Dark_Zurich_val_anon.zip -d ' + str(ds_path))
    os.system('rm Dark_Zurich_val_anon.zip')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'Dark_Zurich'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['val']:
        mask_dir = Path(ds_path) / f'gt/{split}/night/GOPR0356'
        anno_dir = Path(ds_path) / 'annotations_detectron2' / split
        anno_dir.mkdir(parents=True, exist_ok=True)

        # convert the masks to detectron2 format
        for mask_path in tqdm.tqdm(list(mask_dir.glob('*labelIds.png'))):
            mask = np.array(Image.open(mask_path))
            # invalid pixels are marked with 255
            invalid_mask = np.array(Image.open(str(mask_path).replace('labelIds', 'invGray')))
            mask[invalid_mask == 255] = 255
            # convert to ids
            mask = np.vectorize(labels_to_id.get)(mask)
            # save the mask
            Image.fromarray(mask.astype(np.uint8)).save(anno_dir / mask_path.name)

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
