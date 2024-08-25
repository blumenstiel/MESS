
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
    'background', 'candy', 'egg tart', 'french fries', 'chocolate', 'biscuit', 'popcorn', 'pudding', 'ice cream',
    'cheese butter', 'cake', 'wine', 'milkshake', 'coffee', 'juice', 'milk', 'tea', 'almond', 'red beans',
    'cashew', 'dried cranberries', 'soy', 'walnut', 'peanut', 'egg', 'apple', 'date', 'apricot', 'avocado',
    'banana', 'strawberry', 'cherry', 'blueberry', 'raspberry', 'mango', 'olives', 'peach', 'lemon', 'pear',
    'fig', 'pineapple', 'grape', 'kiwi', 'melon', 'orange', 'watermelon', 'steak', 'pork', 'chicken duck',
    'sausage', 'fried meat', 'lamb', 'sauce', 'crab', 'fish', 'shellfish', 'shrimp', 'soup', 'bread', 'corn',
    'hamburg', 'pizza', 'hanamaki baozi', 'wonton dumplings', 'pasta', 'noodles', 'rice', 'pie', 'tofu',
    'eggplant', 'potato', 'garlic', 'cauliflower', 'tomato', 'kelp', 'seaweed', 'spring onion', 'rape', 'ginger',
    'okra', 'lettuce', 'pumpkin', 'cucumber', 'white radish', 'carrot', 'asparagus', 'bamboo shoots', 'broccoli',
    'celery stick', 'cilantro mint', 'snow peas', 'cabbage', 'bean sprouts', 'onion', 'pepper', 'green beans',
    'French beans', 'king oyster mushroom', 'shiitake', 'enoki mushroom', 'oyster mushroom', 'white button mushroom',
    'salad', 'other ingredients'
]


def register_dataset(root):
    ds_name = 'foodseg103'
    root = os.path.join(root, 'FoodSeg103')

    for split, image_dirname, sem_seg_dirname, class_names in [
        ('test', 'Images/img_dir/test', 'Images/ann_dir/test', CLASSES),
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
            classes_of_interest=list(range(1, len(class_names))),
            background_class=0,
        )


_root = os.getenv('DETECTRON2_DATASETS', 'datasets')
register_dataset(_root)
