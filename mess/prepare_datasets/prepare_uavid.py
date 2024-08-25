
# run python mess/prepare_datasets/prepare_uavid.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image

# The 8 classes and corresponding label color (R,G,B) are as follows:
# Background clutter       	(0,0,0)
# Building			(128,0,0)
# Road				(128,64,128)
# Tree				(0,128,0)
# Low vegetation		        (128,128,0)
# Moving car			(64,0,128)
# Static car			(192,0,192)
# Human				(64,64,0)


def download_dataset(dataset_dir, ds_path):
    """
    Downloads the dataset
    """
    ds_path.mkdir(exist_ok=True, parents=True)
    if Path('uavid_v1.5_official_release_image.zip').exists():
        file_path = 'uavid_v1.5_official_release_image.zip'
    elif (dataset_dir / 'uavid_v1.5_official_release_image.zip').exists():
        file_path = dataset_dir / 'uavid_v1.5_official_release_image.zip'
    else:
        raise Exception(f'Dataset and zip not found. '
                        f'Please download validation set from https://uavid.nl and place it root or in {dataset_dir}')
    print('Unzip dataset...')
    os.system(f'unzip {file_path} -d {ds_path}')
    os.system(f'rm {file_path}')

    # remove uavid_v1.5_official_release_image folder from structure
    if os.path.isdir(ds_path / 'uavid_v1.5_official_release_image'):
        os.system(f'mv {ds_path / "uavid_v1.5_official_release_image" / "*"} {ds_path}')
        os.system(f'rmdir {ds_path / "uavid_v1.5_official_release_image"}')


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
    ds_path = dataset_dir / 'uavid_v1.5'
    if not ds_path.exists():
        download_dataset(dataset_dir, ds_path)

    assert (ds_path / 'uavid_val').exists(), f'Dataset or validation set not found in {ds_path}. ' \
                                             'Please download validation set from https://uavid.nl.'

    for split in ['val']:

        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        for img_path in tqdm.tqdm(list((ds_path / ('uavid_' + split)).glob('*/Images/*.png'))):
            id = f'{img_path.parent.parent.stem}_{img_path.stem}'
            # Save tiles
            img = Image.open(img_path)
            img = img.convert('RGB')
            tiles = get_tiles(img, padding=0)
            for i, tile in enumerate(tiles):
                Image.fromarray(tile).save(img_dir / f'{id}_{i}.png')

            # Open mask
            mask = Image.open(str(img_path).replace('Images', 'Labels'))
            # Edit annotations
            # Map RGB values to class index by converting to grayscale and applying a lookup table
            mask = np.array(mask.convert('L'))
            color_to_class = {0: 0,  34: 5, 38: 1,  57: 7,  75: 3,  79: 6, 90: 2, 113: 4}
            mask = np.vectorize(color_to_class.get)(mask)
            # Save mask tiles
            tiles = get_tiles(mask, padding=255)
            for i, tile in enumerate(tiles):
                Image.fromarray(tile).save(anno_dir / f'{id}_{i}.png')

        print(f'Saved {split} set of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
