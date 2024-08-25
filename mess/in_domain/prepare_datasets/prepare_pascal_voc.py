# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Code based on https://github.com/MendelXu/zsseg.baseline/blob/master/datasets/prepare_voc_sem_seg.py
# run: python mess/prepare_datasets/prepare_pascal_voc.py

import os
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image


clsID_to_trID = {
    0: 255,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    255: 255,
}

clsID_to_trID_w_background = clsID_to_trID.copy()
clsID_to_trID_w_background[0] = 20


def convert_to_trainID(
    maskpath, out_mask_dir, is_train, clsID_to_trID=clsID_to_trID, suffix="",
):
    mask = np.array(Image.open(maskpath))
    mask = np.vectorize(clsID_to_trID.get)(mask).astype(np.uint8)

    seg_filename = (
        os.path.join(out_mask_dir, "train" + suffix, os.path.basename(maskpath))
        if is_train
        else os.path.join(out_mask_dir, "val" + suffix, os.path.basename(maskpath))
    )
    if len(np.unique(mask)) == 1 and (np.unique(mask)[0] == 255 or np.unique(mask)[0] == 20):
        # ignore images with only background
        return
    Image.fromarray(mask).save(seg_filename, "PNG")


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading images
    os.system('wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar')
    os.system(f'tar -xvf VOCtrainval_11-May-2012.tar -C {ds_path.parent.parent}')
    os.system(f'rm VOCtrainval_11-May-2012.tar')
    # Downloading annotations
    os.system('wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip')
    os.system('unzip SegmentationClassAug.zip -d ' + str(ds_path))
    os.system('rm SegmentationClassAug.zip')


def main():
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    voc_path = dataset_dir / "VOCdevkit" / "VOC2012"

    if not voc_path.exists():
        download_dataset(voc_path)

    out_mask_dir = voc_path / "annotations_detectron2"

    for name in ["val"]:
        os.makedirs((out_mask_dir / name), exist_ok=True)
        os.makedirs((out_mask_dir / (name + '_bg')), exist_ok=True)

        val_list = [
            os.path.join(voc_path, "SegmentationClassAug", f + ".png")
            for f in np.loadtxt(os.path.join(voc_path, "ImageSets/Segmentation/val.txt"), dtype=np.str).tolist()
        ]

        for file in tqdm.tqdm(val_list):
            convert_to_trainID(file, out_mask_dir, is_train=False)
            convert_to_trainID(file, out_mask_dir, is_train=False,
                               clsID_to_trID=clsID_to_trID_w_background, suffix="_bg")


if __name__ == "__main__":
    main()
