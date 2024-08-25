
try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
except:
    from mess.utils.catalog import DatasetCatalog, MetadataCatalog

from mmseg.datasets import build_dataset
from pathlib import Path


def build_mmseg_dataset(cfg):
    """
    Replaces the build_dataset method from mmseg.datasets
    :param cfg: cfg.type should match the name of the registered Detectron2 dataset
    :return: MMSeg CustomDataset
    """
    # Load Detectron2 Dataset
    dataset_name = cfg['type']
    detectron2_dataset = DatasetCatalog.get(dataset_name)
    detectron2_metadata = MetadataCatalog.get(dataset_name)

    # Get split file
    dataset_dir = Path(detectron2_metadata.image_root).parent
    split_file = str(dataset_dir / dataset_name) + '.txt'
    if not Path(split_file).exists():
        # create file with instances
        with open(split_file, 'w') as f:
            for instance in detectron2_dataset:
                f.write(instance['file_name'].split('/')[-1][:-4])
                f.write('\n')

    # Update MMSeg Config based on Detectron2 Dataset
    cfg['type'] = 'CustomDataset'
    cfg['data_root'] = ''
    cfg['split'] = split_file
    cfg['img_dir'] = detectron2_metadata.image_root
    cfg['ann_dir'] = detectron2_metadata.sem_seg_root
    cfg['img_suffix'] = detectron2_dataset[0]['file_name'][-4:]
    cfg['seg_map_suffix'] = detectron2_dataset[0]['sem_seg_file_name'][-4:]
    cfg['ignore_index'] = detectron2_metadata.ignore_label
    cfg['classes'] = detectron2_metadata.stuff_classes
    cfg['palette'] = [[i,i,i] for i in range(len(detectron2_metadata.stuff_classes))]

    # Load and return MMSeg Dataset
    dataset = build_dataset(cfg)
    return dataset


def get_class_names(name):
    return MetadataCatalog.get(name).stuff_classes


def get_detectron2_datasets():
    return list(DatasetCatalog.data)
