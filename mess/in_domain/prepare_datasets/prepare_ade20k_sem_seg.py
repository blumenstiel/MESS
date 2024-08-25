# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Code adapted from ovseg/datasets/prepare_ade20k_sem_seg.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image


def convert(input, output, index=None):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    if index is not None:
        mapping = {i: k for k, i in enumerate(index)}
        img = np.vectorize(lambda x: mapping[x] if x in mapping else 255)(
            img.astype(np.float)
        ).astype(np.uint8)
    Image.fromarray(img).save(output)


def download_dataset(dataset_dir):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading zip
    # http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    os.system('wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip')
    os.system('unzip ADEChallengeData2016.zip -d ' + str(dataset_dir))
    os.system('rm ADEChallengeData2016.zip')


def main():
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    ds_path = dataset_dir / "ADEChallengeData2016"
    if not ds_path.exists():
        download_dataset(dataset_dir)

    assert dataset_dir.exists(), f"Please download ADE20K dataset to {ds_path} from http://sceneparsing.csail.mit.edu/"

    print('Caution: we only generate the validation set!')
    for name in ["validation"]:
        annotation_dir = ds_path / "annotations" / name
        output_dir = ds_path / "annotations_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file)


if __name__ == "__main__":
    main()
