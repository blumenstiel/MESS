# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import tqdm
import os
import os.path as osp
from pathlib import Path
import gdown

import numpy as np
from PIL import Image
import scipy.io

def convert_pc459(mask_path, new_mask_path):
    mat = scipy.io.loadmat(mask_path)
    mask = mat['LabelMap']
    mask = mask - 1
    min_value = np.amin(mask)
    assert min_value >= 0, print(min_value)
    Image.fromarray(mask).save(new_mask_path, "TIFF")


def download_dataset(dataset_dir, ds_path):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading images
    # http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
    os.system('wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar')
    os.system(f'tar -xvf VOCtrainval_03-May-2010.tar -C {dataset_dir}')
    os.system(f'rm VOCtrainval_03-May-2010.tar')
    # Downloading val split from Google Drive
    # https://drive.google.com/file/d/1BCbiOKtLvozjVnlTJX51koIveUZHCcUh/view
    gdown.download(id='1BCbiOKtLvozjVnlTJX51koIveUZHCcUh', output=str(ds_path / 'pascalcontext_val.txt'))
    # Downloading 459 labels
    os.system('wget https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz')
    os.system(f'tar -xvzf trainval.tar.gz -C {ds_path}')
    os.system(f'rm trainval.tar.gz')
    # Download pc 59 label names
    os.system(f'wget https://codalabuser.blob.core.windows.net/public/trainval_merged.json -P {ds_path}')


def main():
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    print('Caution: we only generate the validation set!')
    pc_path = dataset_dir / "VOCdevkit/VOC2010"

    val_list = open(pc_path / "pascalcontext_val.txt", "r")
    pc459_labels = open(pc_path / "labels.txt", "r")

    if not pc_path.exists():
        download_dataset(dataset_dir, pc_path)

    assert pc_path.exists(), f"Please download Pascal Context dataset to {pc_path}"

    pc459_dict = {}
    for line in pc459_labels.readlines():
        if ':' in line:
            idx, name = line.split(':')
            idx = int(idx.strip())
            name = name.strip()
            pc459_dict[name] = idx

    pc459_dir = pc_path / "annotations_detectron2" / "pc459_val"
    pc459_dir.mkdir(parents=True, exist_ok=True)

    for line in tqdm.tqdm(val_list.readlines()):
        fileid = line.strip()
        ori_mask = f'{pc_path}/trainval/{fileid}.mat'
        pc459_dst = f'{pc459_dir}/{fileid}.tif'
        if osp.exists(ori_mask):
            convert_pc459(ori_mask, pc459_dst)

if __name__ == '__main__':
    main()
