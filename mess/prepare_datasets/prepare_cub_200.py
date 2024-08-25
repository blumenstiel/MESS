
# run python mess/prepare_datasets/prepare_cub_200.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image


def download_dataset(dataset_dir):
    """
    Downloads the dataset
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)
    # Downloading data
    os.system("wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz")
    os.system(f"tar -xvzf CUB_200_2011.tgz -C {dataset_dir}")
    os.system(f"rm CUB_200_2011.tgz")

    # Downloading segmentation masks
    os.system("wget https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz")
    os.system(f"tar -xvzf segmentations.tgz -C {dataset_dir / 'CUB_200_2011'}")
    os.system(f"rm segmentations.tgz")

    # Move attributes to folder
    os.system(f"mv {dataset_dir / 'attributes.txt'} {dataset_dir / 'CUB_200_2011'}")


def main():
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    ds_path = dataset_dir / "CUB_200_2011"
    if not ds_path.exists():
        download_dataset(dataset_dir)

    assert ds_path.exists(), f"Dataset not found in {ds_path}"

    # read image file names
    with open(ds_path / 'images.txt', 'r') as f:
        img_files = [i.split(' ')[1] for i in f.read().splitlines()]

    # read test image list
    with open(ds_path / 'train_test_split.txt', 'r') as f:
        test_images = [not bool(int(i.split(' ')[1])) for i in f.read().splitlines()]

    for split in ['test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        anno_dir = ds_path / 'annotations_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        if split == 'test':
            img_files = np.array(img_files)[test_images]
        else:
            img_files = np.array(img_files)[~np.array(test_images)]

        # iterate over all image files
        for img_file in tqdm.tqdm(img_files):
            img_name = img_file.split('/')[-1]
            # Copy image
            img = Image.open(ds_path / 'images' / img_file)
            img = img.convert('RGB')
            img.save(img_dir / img_name)

            # Open mask
            img_name = img_name.replace('jpg', 'png')
            mask = Image.open(str(ds_path / 'segmentations' / img_file.replace('jpg', 'png'))).convert('L')

            # Edit annotations
            # Using majority voting from 5 labelers to get binary mask
            bin_mask = np.uint8(np.array(mask) > 128)
            # Replace mask with class index
            class_idx = int(img_file.split('.')[0])
            mask = bin_mask * class_idx
            # Save normal mask
            Image.fromarray(mask, 'L').save(anno_dir / img_name, "PNG")

        print(f'Saved {split} images and masks of {ds_path.name} dataset')


if __name__ == "__main__":
    main()
