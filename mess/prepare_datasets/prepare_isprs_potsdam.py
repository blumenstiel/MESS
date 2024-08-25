
# run python mess/prepare_datasets/prepare_isprs_potsdam.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image

# Impervious surfaces (RGB: 255, 255, 255)
# Building (RGB: 0, 0, 255)
# Low vegetation (RGB: 0, 255, 255)
# Tree (RGB: 0, 255, 0)
# Car (RGB: 255, 255, 0)
# Clutter/background (RGB: 255, 0, 0)

class_dict = {
    0: [255, 255, 255],
    1: [0, 0, 255],
    2: [0, 255, 255],
    3: [0, 255, 0],
    4: [255, 255, 0],
    5: [255, 0, 0],
}

test_set = [
    'top_potsdam_2_13',
    'top_potsdam_2_14',
    'top_potsdam_3_13',
    'top_potsdam_3_14',
    'top_potsdam_4_13',
    'top_potsdam_4_14',
    'top_potsdam_4_15',
    'top_potsdam_5_13',
    'top_potsdam_5_14',
    'top_potsdam_5_15',
    'top_potsdam_6_13',
    'top_potsdam_6_14',
    'top_potsdam_6_15',
    'top_potsdam_7_13',
]


def unzip_dataset(dataset_dir, ds_path):
    """
    Downloads the dataset
    """
    ds_path.mkdir(exist_ok=True, parents=True)
    if Path('Potsdam.zip').exists():
        file_path = 'Potsdam.zip'
    elif (dataset_dir / 'Potsdam.zip').exists():
        file_path = dataset_dir / 'Potsdam.zip'
    else:
        raise Exception('Zip not found. Please download the Potsdam dataset from '
                        'https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx')
    # Downloading zip
    print('Unzip dataset...')
    os.system(f'unzip {file_path} -d {ds_path}')
    os.system(f'unzip {ds_path / "Potsdam" / "2_Ortho_RGB.zip"} -d {ds_path}')
    os.system(f'unzip {ds_path / "Potsdam" / "3_Ortho_IRRG.zip"} -d {ds_path}')
    os.system(f'unzip {ds_path / "Potsdam" / "5_Labels_all.zip"} -d {ds_path / "Labels"}')
    os.system(f'rm -r {ds_path / "Potsdam"}')
    os.system(f'rm {file_path}')


def get_tiles(input, h_size=1024, w_size=1024, padding=0):
    input = np.array(input)
    h, w = input.shape[:2]
    tiles = []
    for i in range(0, h, h_size):
        for j in range(0, w, w_size):
            tile = input[i:i + h_size, j:j + w_size]
            if tile.shape[:2] == [h_size, w_size]:
                tiles.append(tile)
            else:
                # padding
                if len(tile.shape) == 2:
                    # Mask (2 channels, padding with ignore_value)
                    padded_tile = np.ones((h_size, w_size), dtype=np.uint8) * padding
                else:
                    # RGB (3 channels, padding usually 0)
                    padded_tile = np.ones((h_size, w_size, tile.shape[2]), dtype=np.uint8) * padding
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tiles.append(padded_tile)
    return tiles


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'ISPRS_Potsdam'
    if not ds_path.exists():
        unzip_dataset(dataset_dir, ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(anno_dir, exist_ok=True)
        for dir in ['rgb', 'irrg']:
            os.makedirs(img_dir / dir, exist_ok=True)

        for img_path in tqdm.tqdm((ds_path / '2_Ortho_RGB').glob('*.tif')):
            id = img_path.stem[:-4]
            if (id not in test_set and split == 'test') or (id in test_set and split != 'test'):
                continue

            # RGB images
            img = Image.open(img_path).convert('RGB')
            tiles = get_tiles(img, padding=0)
            for i, tile in enumerate(tiles):
                Image.fromarray(tile).save(img_dir / 'rgb' / f'{id}_{i}.png')

            # IRRG images
            img = Image.open(str(img_path).replace('2_Ortho_', '3_Ortho_').replace('RGB', 'IRRG'))
            tiles = get_tiles(img, padding=0)
            for i, tile in enumerate(tiles):
                Image.fromarray(tile).save(img_dir / 'irrg' / f'{id}_{i}.png')

            # Masks
            mask = Image.open(str(img_path).replace('2_Ortho_RGB', 'Labels').replace('RGB', 'label'))
            # Binarize mask because some images have gradings
            mask = (np.array(mask) > 128).astype(np.uint8) * 255
            # Edit annotations
            # Map RGB values to class index by converting to grayscale and applying a lookup table
            for class_idx, rgb in class_dict.items():
                mask[(mask == rgb).all(axis=-1)] = class_idx
            mask = mask[:, :, 0]

            # Save mask
            tiles = get_tiles(mask, padding=255)
            for i, tile in enumerate(tiles):
                Image.fromarray(tile).save(anno_dir / f'{id}_{i}.png')

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()