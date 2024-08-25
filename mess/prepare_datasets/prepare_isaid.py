
# run python mess/prepare_datasets/prepare_isaid.py

import os
import tqdm
import gdown
import numpy as np
from pathlib import Path
from PIL import Image


# iSAID dataset color to class mapping
color_to_class = {0: [0, 0, 0],  # unlabeled
                  1: [0, 0, 63],  # ship
                  2: [0, 63, 63],  # storage_tank
                  3: [0, 63, 0],  # baseball_diamond
                  4: [0, 63, 127],  # tennis_court
                  5: [0, 63, 191],  # basketball_court
                  6: [0, 63, 255],  # Ground_Track_Field
                  7: [0, 127, 63],  # Bridge
                  8: [0, 127, 127],  # Large_Vehicle
                  9: [0, 0, 127],  # Small_Vehicle
                  10: [0, 0, 191],  # Helicopter
                  11: [0, 0, 255],  # Swimming_pool
                  12: [0, 191, 127],  # Roundabout
                  13: [0, 127, 191],  # Soccer_ball_field
                  14: [0, 127, 255],  # plane
                  15: [0, 100, 155],  # Harbor
                  }


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    ds_path.mkdir(exist_ok=True, parents=True)
    print('Downloading dataset...')
    # Download from Google Drive
    # Download val images https://drive.google.com/drive/folders/1RV7Z5MM4nJJJPUs6m9wsxDOJxX6HmQqZ?usp=share_link4
    gdown.download_folder(id='1RV7Z5MM4nJJJPUs6m9wsxDOJxX6HmQqZ', output=str(ds_path))
    os.system(f'unzip {ds_path / "part1.zip"} -d {ds_path / "val"}')
    os.system(f'rm {ds_path / "part1.zip"}')

    # Download val mask https://drive.google.com/drive/folders/1jlVr4ClmeBA01IQYx7Aq3Scx2YS1Bmpb
    gdown.download_folder(id='1jlVr4ClmeBA01IQYx7Aq3Scx2YS1Bmpb', output=str(ds_path))
    os.system(f'unzip {ds_path / "images.zip"} -d {ds_path / "raw_val"}')
    os.system(f'rm {ds_path / "images.zip"}')


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
    ds_path = dataset_dir / 'isaid'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    for split in ['val']:
        assert (ds_path / f'raw_{split}').exists(), f'Raw {split} images not found in {ds_path / f"raw_{split}"}'
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        # Convert annotations to detectron2 format
        for mask_path in tqdm.tqdm(sorted((ds_path / f'raw_{split}' / 'images').glob('*.png'))):
            file = mask_path.name
            id = file.split('_')[0]
            # Open image
            img = Image.open(ds_path / split / 'images' / f'{id}.png')
            # Open mask
            mask = np.array(Image.open(mask_path))
            # Map RGB values to class index by applying a lookup table
            for class_idx, rgb in color_to_class.items():
                mask[(mask == rgb).all(axis=-1)] = class_idx
            mask = mask[:, :, 0]

            # Get tiles
            img_tiles = get_tiles(img, padding=0)
            mask_tiles = get_tiles(mask, padding=255)
            # Save tiles
            for i, (img_tile, mask_tile) in enumerate(zip(img_tiles, mask_tiles)):
                Image.fromarray(img_tile).save(img_dir / f'{id}_{i}.png')
                Image.fromarray(mask_tile).save(anno_dir / f'{id}_{i}.png')

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
