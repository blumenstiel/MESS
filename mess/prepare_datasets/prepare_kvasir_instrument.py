
# run python mess/prepare_datasets/prepare_kvasir_instrument.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image


def download_dataset(dataset_dir, ds_path):
    """
    Downloads the dataset
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)
    ds_path.mkdir(exist_ok=True, parents=True)
    print('Downloading dataset...')
    os.system("wget https://datasets.simula.no/downloads/kvasir-instrument.zip")
    os.system("unzip kvasir-instrument.zip -d " + str(dataset_dir))
    os.system("rm kvasir-instrument.zip")

    print('Creating images directory...')
    os.system(f"tar -xvzf {ds_path}/images.tar.gz -C {ds_path}")
    os.system(f"tar -xvzf {ds_path}/masks.tar.gz -C {ds_path}")
    os.system(f"rm {ds_path}/images.tar.gz")
    os.system(f"rm {ds_path}/masks.tar.gz")


def main():
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    ds_path = dataset_dir / "kvasir-instrument"
    if not ds_path.exists():
        download_dataset(dataset_dir, ds_path)

    assert ds_path.exists(), f"Dataset not found in {ds_path / 'images'}"

    for split in ['test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        with open(ds_path / f"{split}.txt", "r") as f:
            ids = [line.rstrip() for line in f]

        for id in tqdm.tqdm(ids):
            img = Image.open(ds_path / f"images/{id}.jpg").convert('RGB')
            img.save(img_dir / f'{id}.png', "PNG")

            mask = Image.open(ds_path / f"masks/{id}.png")
            # 0: others, 1: instrument
            mask = np.uint8(np.array(mask)[:,:,0] / 255)
            Image.fromarray(mask).save(anno_dir / f'{id}.png', "PNG")

        print(f'Saved {split} images and masks of kvasir-instrument dataset')


if __name__ == '__main__':
    main()

