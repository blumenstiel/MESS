
import os
import tqdm
import json
import numpy as np
import gdown
from pathlib import Path
from PIL import Image
from pycocotools import mask as m

_mapping = np.sort(
    np.array([
        0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 23, 397, 25, 284,
        158, 159, 416, 33, 162, 420, 454, 295, 296, 427, 44, 45, 46, 308, 59,
        440, 445, 31, 232, 65, 354, 424, 68, 326, 72, 458, 34, 207, 80, 355,
        85, 347, 220, 349, 360, 98, 187, 104, 105, 366, 189, 368, 113, 115
    ]))
_key = np.array(range(len(_mapping))).astype('uint8')
_key = _key - 1

_map = {}
for (k, v) in zip(_mapping, _key):
    _map[k] = v

def generate_labels(img_id, anno, out_dir):
    def _class_to_index(mask, _map):
        out = np.ones_like(mask, dtype=np.uint8) * 255
        for k, v in _map.items():
            out[mask == k] = v
        return out

    img_id['image_id']
    mask = Image.fromarray(
        _class_to_index(anno, _map))
        #_class_to_index(detail.getMask(img_id), _map))
    filename = img_id['file_name']
    mask.save(os.path.join(out_dir, filename.replace('jpg', 'png')))
    return os.path.splitext(os.path.basename(filename))[0]


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
    voc_path = dataset_dir / "VOCdevkit" / "VOC2010"
    out_dir = voc_path / "annotations_detectron2" / "pc59_val"
    json_path = voc_path / "trainval_merged.json"

    if not voc_path.exists():
        download_dataset(dataset_dir, voc_path)

    assert voc_path.exists(), f"Please download Pascal Context dataset to {voc_path}"


    os.makedirs(out_dir, exist_ok=True)
    img_dir = out_dir / "JPEGImages"

    print("loading annotations...")
    data = json.load(open(json_path, 'r'))
    val_images = {d['image_id'] : d for d in data['images'] if d['phase'] == "val"}
    annos = {}

    print("building annotations...")
    for ann in data['annos_segmentation']:
        key = ann['image_id']
        if key in val_images.keys():
            if key in annos.keys():
                annos[key].append(ann)
            else:
                annos[key] = [ann]

    for k, v in annos.items():
        mask = np.zeros((val_images[k]['height'], val_images[k]['width']))
        for c in v:
            x = m.decode(c['segmentation'])
            mask[np.nonzero(x)] = c['category_id']
        
        annos[k] = mask

    print("converting annotations...")
    for id, dat in tqdm.tqdm(val_images.items()):
        generate_labels(dat, annos[id],out_dir=out_dir)
    
    print("done")


if __name__ == "__main__":
    main()
