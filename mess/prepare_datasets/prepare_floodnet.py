
# run python mess/prepare_datasets/prepare_floodnet.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image


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
    ds_path = dataset_dir / 'FloodNet-Supervised_v1.0'

    assert ds_path.exists(), f'Dataset not found in {ds_path}. Please download from ' \
                             f'https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD'

    for split in ['test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        for img_path in tqdm.tqdm(list((ds_path / 'test' / 'test-org-img').glob('*.jpg'))):
            id = img_path.stem
            # Open image
            img = Image.open(img_path).convert('RGB')
            # Save image tiles
            img_tiles = get_tiles(img, padding=0)
            for i, tile in enumerate(img_tiles):
                Image.fromarray(tile).save(img_dir / f'{id}_{i}.jpg')

            # Open mask
            mask = Image.open(str(img_path).replace('test-org-img', 'test-label-img').replace('.jpg', '_lab.png'))
            # Save mask tiles
            tiles = get_tiles(mask, padding=255)
            for i, tile in enumerate(tiles):
                Image.fromarray(tile).save(anno_dir / f'{id}_{i}.png')

        print(f'Saved {split} set of {ds_path.name} dataset')


if __name__ == '__main__':
    main()
