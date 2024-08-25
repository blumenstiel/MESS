
import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except:
    from mess.utils.catalog import DatasetCatalog, MetadataCatalog
    from mess.utils.data import load_sem_seg
    from mess.utils.colormap import colormap

# TODO: Change class names
CLASSES = [
    'others', 'object'
]


# TODO: Add this script to mess/dataset/__init__.py
def register_dataset(root):
    # TODO: Change dataset name and path
    ds_name = '<DATASET>'
    root = os.path.join(root, '<DATASET>')

    for split, image_dirname, sem_seg_dirname, class_names in [
        # TODO: Add additional splits
        ('test', 'images_detectron2/test', 'annotations_detectron2/test', CLASSES),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        # TODO: Check image and mask (gt) extensions and change if required
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
            # TODO: Change ignore label if not 255 (65536 when tif mask files are used)
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True),
            # TODO: Optionally change classes of interest is used for evaluation
            classes_of_interest=list(range(1, len(class_names))),
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
