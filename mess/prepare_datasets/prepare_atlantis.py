
# run python mess/prepare_datasets/prepare_atlantis.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
color_to_class = {c: i for i, c in enumerate(range(1, 57))}
color_to_class[0] = 255


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    # Dataset page: https://github.com/smhassanerfani/atlantis.git
    print('Downloading dataset...')
    # Downloading github repo
    os.system('git clone https://github.com/smhassanerfani/atlantis.git')
    ds_path.mkdir(exist_ok=True, parents=True)
    # Move images and masks to dataset folder
    os.system('mv atlantis/atlantis ' + str(ds_path))
    # Delete github repo
    os.system('rm -R atlantis')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'atlantis'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        for img_path in tqdm.tqdm((ds_path / 'images' / split).glob('*/*.jpg')):
            # Load and convert image and mask
            img = Image.open(img_path)
            img = img.convert('RGB')
            img.save(img_dir / img_path.name)

            mask = Image.open(str(img_path).replace('images', 'masks').replace('jpg', 'png'))
            # Replace grey values with class index
            mask = np.vectorize(color_to_class.get)(np.array(mask)).astype(np.uint8)
            Image.fromarray(mask).save(anno_dir / img_path.name.replace('jpg', 'png'))

    print(f'Saved images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()