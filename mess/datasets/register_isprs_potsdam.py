import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except:
    from mess.utils.catalog import DatasetCatalog, MetadataCatalog
    from mess.utils.data import load_sem_seg
    from mess.utils.colormap import colormap

CLASSES = (
    'road',
    'building',
    'grass',
    'tree',
    'car',
    'others',
)

CLASSES_OFFICIAL = (
    'imperious surface',
    'building',
    'low vegetation',
    'tree',
    'car',
    'others',
)


def register_dataset(root):
    ds_name = 'isprs_potsdam'
    root = os.path.join(root, 'ISPRS_Potsdam')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('test_rgb', 'images_detectron2/test/rgb', 'annotations_detectron2/test', CLASSES),
        ('test_irrg', 'images_detectron2/test/irrg', 'annotations_detectron2/test', CLASSES),
        ('test_irrg_official', 'images_detectron2/test/irrg', 'annotations_detectron2/test', CLASSES_OFFICIAL),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='png'
            ),
        )
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type='sem_seg',
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True)[[3, 1, 4, 15, 2, 0]],
            classes_of_interest=list(range(len(class_names) - 1)),
            background_class=5,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
