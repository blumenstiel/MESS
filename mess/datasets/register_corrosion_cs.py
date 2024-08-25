
import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except:
    from mess.utils.catalog import DatasetCatalog, MetadataCatalog
    from mess.utils.data import load_sem_seg
    from mess.utils.colormap import colormap

# Classes:
# 0: background
# 1: Fair
# 2: Poor: Corrosion that appears as deeper-than-surface corrosion or beginning to disintegrate portions of steel is considered “3 -Poor” (includes Pack Rust)
# 3: Severe: Corrosion that seems to have disintegrated portions of steel and/or corrosion that seems to affect structural integrity of the bridge component is annotated as “4 -Severe”

CLASSES = (
    'others',
    'steel with fair corrosion',
    'steel with poor corrosion',
    'steel with severe corrosion',
)

CLASSES_OFFICIAL = (
    'others',
    'fair corrosion',
    'poor corrosion',
    'severe corrosion',
)


def register_dataset(root):
    ds_name = 'corrosion_cs'
    root = os.path.join(root, 'Corrosion Condition State Classification')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('test', 'original/Test/images', 'original/Test/masks', CLASSES),
        ('test_official', 'original/Test/images', 'original/Test/masks', CLASSES_OFFICIAL),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f'{ds_name}_sem_seg_{split}'
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext='png', image_ext='jpeg'
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
