import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except:
    from mess.utils.catalog import DatasetCatalog, MetadataCatalog
    from mess.utils.data import load_sem_seg
    from mess.utils.colormap import colormap

PASCAL_VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                      "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                      "tvmonitor"]
PASCAL_VOC_CLASSES_BACKGROUND = PASCAL_VOC_CLASSES + ["background"]


def register_all_pascal_voc(root):
    root = os.path.join(root, "VOCdevkit/VOC2012")
    for name, image_dirname, sem_seg_dirname, class_names in [
        ("val", "JPEGImages", "annotations_detectron2", PASCAL_VOC_CLASSES),
        ("val_bg", "JPEGImages", "annotations_detectron2", PASCAL_VOC_CLASSES_BACKGROUND),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname, name)
        name = f"voc_2012_sem_seg_{name}"

        if name in DatasetCatalog.list():
            # dataset is already registered
            MetadataCatalog.get(name).set(
                stuff_colors=colormap(rgb=True),
                stuff_classes=class_names,
            )
            continue

        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir,
                                      seg_seg_root=gt_dir,
                                      evaluator_type="sem_seg",
                                      ignore_label=255,
                                      stuff_classes=class_names,
                                      stuff_colors=colormap(rgb=True),
                                      classes_of_interest=list(range(0, 20)),
                                      background_class=20,
                                      )


_root = os.getenv("DETECTRON2_DATASETS", "")
register_all_pascal_voc(_root)
