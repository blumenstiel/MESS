
# run python mess/prepare_datasets/prepare_paxray.py

import os
import tqdm
import gdown
import json
import numpy as np
from pathlib import Path
from PIL import Image

# all labels
# {'0': 'lung_overall',
#  '1': 'lung_right',
#  '2': 'lung_left',
#  '3': 'lung_left_lobeupper',
#  '4': 'lung_left_lobelower',
#  '5': 'lung_right_lobeupper',
#  '6': 'lung_right_lobemiddle',
#  '7': 'lung_right_lobelower',
#  '8': 'lung_right_vessel',
#  '9': 'lung_left_vessel',
#  '10': 'mediastinum_overall',
#  '11': 'mediastinum_lower',
#  '12': 'mediastinum_upper',
#  '13': 'mediastinum_anterior',
#  '14': 'mediastinum_middle',
#  '15': 'mediastinum_posterior',
#  '16': 'mediastinum_upper',
#  '17': 'heart',
#  '18': 'airways',
#  '19': 'esophagus',
#  '20': 'aorta',
#  '21': 'aorta_ascending',
#  '22': 'aorta_arch',
#  '23': 'aorta_descending',
#  '24': 'bones',
#  '25': 'spine',
#  '26': 'c1',
#  '27': 'c2',
#  '28': 'c3',
#  '29': 'c4',
#  '30': 'c5',
#  '31': 'c6',
#  '32': 'c7',
#  '33': 't1',
#  '34': 't2',
#  '35': 't3',
#  '36': 't4',
#  '37': 't5',
#  '38': 't6',
#  '39': 't7',
#  '40': 't8',
#  '41': 't9',
#  '42': 't10',
#  '43': 't11',
#  '44': 't12',
#  '45': 'l1',
#  '46': 'l2',
#  '47': 'l3',
#  '48': 'l4',
#  '49': 'l5',
#  '50': 'l6',
#  '51': 'sacrum',
#  '52': 'cocygis',
#  '53': 't13',
#  '54': 'ribs',
#  '55': 'rib_1',
#  '56': 'rib_2',
#  '57': 'rib_3',
#  '58': 'rib_4',
#  '59': 'rib_5',
#  '60': 'rib_6',
#  '61': 'rib_7',
#  '62': 'rib_8',
#  '63': 'rib_9',
#  '64': 'rib_10',
#  '65': 'rib_11',
#  '66': 'rib_12',
#  '67': 'rib_anterior_1',
#  '68': 'rib_posterior_1',
#  '69': 'rib_anterior_2',
#  '70': 'rib_posterior_2',
#  '71': 'rib_anterior_3',
#  '72': 'rib_posterior_3',
#  '73': 'rib_anterior_4',
#  '74': 'rib_posterior_4',
#  '75': 'rib_anterior_5',
#  '76': 'rib_posterior_5',
#  '77': 'rib_anterior_6',
#  '78': 'rib_posterior_6',
#  '79': 'rib_anterior_7',
#  '80': 'rib_posterior_7',
#  '81': 'rib_anterior_8',
#  '82': 'rib_posterior_8',
#  '83': 'rib_anterior_9',
#  '84': 'rib_posterior_9',
#  '85': 'rib_anterior_10',
#  '86': 'rib_posterior_10',
#  '87': 'rib_anterior_11',
#  '88': 'rib_posterior_11',
#  '89': 'rib_anterior_12',
#  '90': 'rib_posterior_12',
#  '91': 'rib_left_1',
#  '92': 'rib_right_1',
#  '93': 'rib_left_2',
#  '94': 'rib_right_2',
#  '95': 'rib_left_3',
#  '96': 'rib_right_3',
#  '97': 'rib_left_4',
#  '98': 'rib_right_4',
#  '99': 'rib_left_5',
#  '100': 'rib_right_5',
#  '101': 'rib_left_6',
#  '102': 'rib_right_6',
#  '103': 'rib_left_7',
#  '104': 'rib_right_7',
#  '105': 'rib_left_8',
#  '106': 'rib_right_8',
#  '107': 'rib_left_9',
#  '108': 'rib_right_9',
#  '109': 'rib_left_10',
#  '110': 'rib_right_10',
#  '111': 'rib_left_11',
#  '112': 'rib_right_11',
#  '113': 'rib_left_12',
#  '114': 'rib_right_12',
#  '115': 'rib_left_anterior_1',
#  '116': 'rib_left_posterior_1',
#  '117': 'rib_left_anterior_2',
#  '118': 'rib_left_posterior_2',
#  '119': 'rib_left_anterior_3',
#  '120': 'rib_left_posterior_3',
#  '121': 'rib_left_anterior_4',
#  '122': 'rib_left_posterior_4',
#  '123': 'rib_left_anterior_5',
#  '124': 'rib_left_posterior_5',
#  '125': 'rib_left_anterior_6',
#  '126': 'rib_left_posterior_6',
#  '127': 'rib_left_anterior_7',
#  '128': 'rib_left_posterior_7',
#  '129': 'rib_left_anterior_8',
#  '130': 'rib_left_posterior_8',
#  '131': 'rib_left_anterior_9',
#  '132': 'rib_left_posterior_9',
#  '133': 'rib_left_anterior_10',
#  '134': 'rib_left_posterior_10',
#  '135': 'rib_left_anterior_11',
#  '136': 'rib_left_posterior_11',
#  '137': 'rib_left_anterior_12',
#  '138': 'rib_left_posterior_12',
#  '139': 'rib_right_anterior_1',
#  '140': 'rib_right_posterior_1',
#  '141': 'rib_right_anterior_2',
#  '142': 'rib_right_posterior_2',
#  '143': 'rib_right_anterior_3',
#  '144': 'rib_right_posterior_3',
#  '145': 'rib_right_anterior_4',
#  '146': 'rib_right_posterior_4',
#  '147': 'rib_right_anterior_5',
#  '148': 'rib_right_posterior_5',
#  '149': 'rib_right_anterior_6',
#  '150': 'rib_right_posterior_6',
#  '151': 'rib_right_anterior_7',
#  '152': 'rib_right_posterior_7',
#  '153': 'rib_right_anterior_8',
#  '154': 'rib_right_posterior_8',
#  '155': 'rib_right_anterior_9',
#  '156': 'rib_right_posterior_9',
#  '157': 'rib_right_anterior_10',
#  '158': 'rib_right_posterior_10',
#  '159': 'rib_right_anterior_11',
#  '160': 'rib_right_posterior_11',
#  '161': 'rib_right_anterior_12',
#  '162': 'rib_right_posterior_12',
#  '163': 'diaphragm',
#  '164': 'hemidiaphragm_right',
#  '165': 'hemidiaphragm_left'}

def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    ds_path.mkdir(exist_ok=True, parents=True)
    print('Downloading dataset...')
    # Download from Google Drive
    # https://drive.google.com/file/d/19HPPhKf9TDv4sO3UV-nI3Jhi4nCv_Zyc/view?usp=share_link
    gdown.download(f"https://drive.google.com/uc?export=download&confirm=pbef&id=19HPPhKf9TDv4sO3UV-nI3Jhi4nCv_Zyc")
    os.system('unzip paxray_dataset.zip -d ' + str(ds_path))
    os.system('rm paxray_dataset.zip')


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'paxray_dataset'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    # get label dictionary and split ids
    with open(ds_path / 'paxray.json', 'r') as f:
        data = json.load(f)

    # binary predictions because of overlapping masks
    target_labels = {
        0: 'lungs',
        10: 'mediastinum',
        24: 'bones',
        163: 'diaphragm',
    }

    for split in ['test']:
        # create directories
        img_dir = ds_path / 'images_detectron2' / split
        os.makedirs(img_dir, exist_ok=True)
        anno_dir = ds_path / 'annotations_detectron2' / split
        for label in target_labels.values():
            (anno_dir / label).mkdir(parents=True, exist_ok=True)

        for paths in tqdm.tqdm(data[split]):
            # Copy image
            img = Image.open(ds_path / paths['image'])
            img = img.convert('RGB')
            img.save(img_dir / paths['image'][7:])

            # Open mask from .npy file
            mask = np.load(ds_path / paths['target'])
            # Save masks of each label separately for binary predictions
            for idx, label in target_labels.items():
                Image.fromarray(mask[idx].astype(np.uint8)).save(anno_dir / label / paths['image'][7:])

        print(f'Saved {split} images and masks for {", ".join(target_labels.values())} of {ds_path.name} dataset')

        # delete original labels to save disk space
        os.system(f'rm -r {ds_path / "labels"}')
        print(f'Deleted original labels of {ds_path.name} dataset to save disk space')


if __name__ == '__main__':
    main()
