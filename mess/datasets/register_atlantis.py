
import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except:
    from mess.utils.catalog import DatasetCatalog, MetadataCatalog
    from mess.utils.data import load_sem_seg
    from mess.utils.colormap import colormap

CLASSES = [
    'bicycle',
    'boat',
    'breakwater',
    'bridge',
    'building',
    'bus',
    'canal',
    'car',
    'cliff',
    'culvert',
    'cypress tree',
    'dam',
    'ditch',
    'fence',
    'fire hydrant',
    'fjord',
    'flood',
    'glaciers',
    'hot spring',
    'lake',
    'levee',
    'lighthouse',
    'mangrove',
    'marsh',
    'motorcycle',
    'offshore platform',
    'parking meter',
    'person',
    'pier',
    'pipeline',
    'pole',
    'puddle',
    'rapids',
    'reservoir',
    'river',
    'river delta',
    'road',
    'sea',
    'ship',
    'shoreline',
    'sidewalk',
    'sky',
    'snow',
    'spillway',
    'swimming pool',
    'terrain',
    'traffic sign',
    'train',
    'truck',
    'umbrella',
    'vegetation',
    'wall',
    'water tower',
    'water well',
    'waterfall',
    'wetland'
]


def register_dataset(root):
    ds_name = 'atlantis'
    root = os.path.join(root, 'atlantis')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('test', 'images_detectron2/test', 'annotations_detectron2/test', CLASSES),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='jpg'
            ),
        )
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type='sem_seg',
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True),
            classes_of_interest=list(range(0, len(class_names))),
            background_class=255,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
