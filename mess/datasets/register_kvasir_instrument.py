
import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except:
    from mess.utils.catalog import DatasetCatalog, MetadataCatalog
    from mess.utils.data import load_sem_seg
    from mess.utils.colormap import colormap

CLASSES = ('others', 'tool')

CLASSES_OFFICIAL = ('others', 'surgical instrument')

def register_kvasir_instrument(root):
    root = os.path.join(root, "kvasir-instrument")

    for name, image_dirname, sem_seg_dirname, class_names in [
        ("test", "images_detectron2/test", "annotations_detectron2/test", CLASSES),
        ("test_official", "images_detectron2/test", "annotations_detectron2/test", CLASSES_OFFICIAL),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"kvasir_instrument_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="png"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True),
            classes_of_interest=[1],
            background_class=0,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_kvasir_instrument(_root)
