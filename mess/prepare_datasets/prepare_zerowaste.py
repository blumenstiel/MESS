
# run python mess/prepare_datasets/prepare_zerowaste.py

import os
from pathlib import Path


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    ds_path.mkdir(exist_ok=True, parents=True)
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://zenodo.org/record/6412647/files/zerowaste-f-final.zip')
    os.system('unzip zerowaste-f-final.zip -d ' + str(ds_path))
    os.system('rm zerowaste-f-final.zip')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'zerowaste-f'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    print(f'Saved images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
