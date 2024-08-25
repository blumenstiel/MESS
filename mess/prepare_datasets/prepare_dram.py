
# run python mess/prepare_datasets/prepare_dram.py

import tqdm
import os
import numpy as np
from pathlib import Path
from PIL import Image

vocId_to_classId = {
    0: 11,  # background
    1: 11,  # aeroplane
    2: 11,  # bicycle
    3: 0,  # bird
    4: 1,  # boat
    5: 2,  # bottle
    6: 11,  # bus
    7: 11,  # car
    8: 3,  # cat
    9: 4,  # chair
    10: 5,  # cow
    11: 11,  # diningtable
    12: 6,  # dog
    13: 7,  # horse
    14: 11,  # motorbike
    15: 8,  # person
    16: 9,  # pottedplant
    17: 10,  # sheep
    18: 11,  # sofa
    19: 11,  # train
    20: 11,  # tvmonitor
    255: 255,  # ignore value
}


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    # Dataset page: https://faculty.runi.ac.il/arik/site/artseg/Dram-Dataset.html
    print('Downloading dataset...')
    # Downloading zip
    os.system('wget https://faculty.runi.ac.il/arik/site/artseg/DRAM_processed.zip')
    ds_path.mkdir(exist_ok=True, parents=True)
    os.system('unzip DRAM_processed.zip -d ' + str(ds_path))
    os.system(f'cd {ds_path} && unrar x DRAM_processed.rar')
    os.system('rm DRAM_processed.zip')
    os.system('rm ' + str(ds_path / 'DRAM_processed.rar'))


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'DRAM_processed'
    if not ds_path.exists():
        download_dataset(dataset_dir)

    assert ds_path.exists(), f'Dataset not found in {ds_path}. ' \
                             f'You may need to install unrar for extracting the dataset by running:\n' \
                             f'sudo apt install unrar\n' \
                             f'or download and extract the dataset manually from: \n' \
                             f'https://faculty.runi.ac.il/arik/site/artseg/DRAM_processed.zip' \

    # train images do not have labels
    for split in ['test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        for img_path in tqdm.tqdm((ds_path / split).glob('**/*.jpg')):
            # Load and convert image and mask
            img = Image.open(img_path)
            img = img.convert('RGB')
            # Add artist name to handle avoid duplicate file names
            img.save(img_dir / (img_path.parent.name + '_' + img_path.name))

            mask = Image.open(str(img_path).replace(split, 'labels').replace('.jpg', '.png'))
            # Convert to detectron2 format
            mask = np.vectorize(vocId_to_classId.get)(np.array(mask)).astype(np.uint8)
            Image.fromarray(mask).save(anno_dir / (img_path.parent.name + '_' + img_path.name.replace('.jpg', '.png')))

    print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
