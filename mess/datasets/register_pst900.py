
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
    'background',
    'fire extinguisher',
    'backpack',
    'drill',
    'human',
)

CLASSES_OFFICAL = (
    'background',
    'fire extinguisher',
    'backpack',
    'drill',
    'rescue randy',
)

def register_dataset(root):
    ds_name = 'pst900'
    root = os.path.join(root, 'PST900_RGBT_Dataset')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('test', 'test/thermal', 'test/labels', CLASSES),
        ('test_official', 'test/thermal', 'test/labels', CLASSES_OFFICAL),
        ('test_pseudo', 'test/thermal_pseudo', 'test/labels', CLASSES),
        ('test_rgb', 'test/rgb', 'test/labels', CLASSES),
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
            stuff_colors=colormap(rgb=True),
            classes_of_interest=list(range(1, len(class_names))),
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
