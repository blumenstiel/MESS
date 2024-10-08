import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except:
    from mess.utils.catalog import DatasetCatalog, MetadataCatalog
    from mess.utils.data import load_sem_seg
    from mess.utils.colormap import colormap

CLASSES = ['background',
           'Black-footed Albatross',
           'Laysan Albatross',
           'Sooty Albatross',
           'Groove billed Ani',
           'Crested Auklet',
           'Least Auklet',
           'Parakeet Auklet',
           'Rhinoceros Auklet',
           'Brewer Blackbird',
           'Red winged Blackbird',
           'Rusty Blackbird',
           'Yellow-headed Blackbird',
           'Bobolink',
           'Indigo Bunting',
           'Lazuli Bunting',
           'Painted Bunting',
           'Cardinal',
           'Spotted Catbird',
           'Gray Catbird',
           'Yellow-breasted Chat',
           'Eastern Towhee',
           'Chuck will Widow',
           'Brandt Cormorant',
           'Red-faced Cormorant',
           'Pelagic Cormorant',
           'Bronzed Cowbird',
           'Shiny Cowbird',
           'Brown Creeper',
           'American Crow',
           'Fish Crow',
           'Black-billed Cuckoo',
           'Mangrove Cuckoo',
           'Yellow-billed Cuckoo',
           'Gray-crowned Rosy-Finch',
           'Purple Finch',
           'Northern Flicker',
           'Acadian Flycatcher',
           'Great Crested Flycatcher',
           'Least Flycatcher',
           'Olive-sided Flycatcher',
           'Scissor-tailed Flycatcher',
           'Vermilion Flycatcher',
           'Yellow-bellied Flycatcher',
           'Frigatebird',
           'Northern Fulmar',
           'Gadwall',
           'American Goldfinch',
           'European Goldfinch',
           'Boat-tailed Grackle',
           'Eared Grebe',
           'Horned Grebe',
           'Pied-billed Grebe',
           'Western Grebe',
           'Blue Grosbeak',
           'Evening Grosbeak',
           'Pine Grosbeak',
           'Rose-breasted Grosbeak',
           'Pigeon Guillemot',
           'California Gull',
           'Glaucous-winged Gull',
           'Heermann Gull',
           'Herring Gull',
           'Ivory Gull',
           'Ring-billed Gull',
           'Slaty-backed Gull',
           'Western Gull',
           'Anna Hummingbird',
           'Ruby-throated Hummingbird',
           'Rufous Hummingbird',
           'Green Violetear',
           'Long-tailed Jaeger',
           'Pomarine Jaeger',
           'Blue Jay',
           'Florida Jay',
           'Green Jay',
           'Dark-eyed Junco',
           'Tropical Kingbird',
           'Gray Kingbird',
           'Belted Kingfisher',
           'Green Kingfisher',
           'Pied Kingfisher',
           'Ringed Kingfisher',
           'White-breasted Kingfisher',
           'Red-legged Kittiwake',
           'Horned Lark',
           'Pacific Loon',
           'Mallard',
           'Western Meadowlark',
           'Hooded Merganser',
           'Red-breasted Merganser',
           'Mockingbird',
           'Nighthawk',
           'Clark Nutcracker',
           'White-breasted Nuthatch',
           'Baltimore Oriole',
           'Hooded Oriole',
           'Orchard Oriole',
           'Scott Oriole',
           'Ovenbird',
           'Brown Pelican',
           'White Pelican',
           'Western Wood-Pewee',
           'Sayornis',
           'American Pipit',
           'Whip-poor-Will',
           'Horned Puffin',
           'Common Raven',
           'White-necked Raven',
           'American Redstart',
           'Geococcyx',
           'Loggerhead Shrike',
           'Great Grey Shrike',
           'Baird Sparrow',
           'Black-throated Sparrow',
           'Brewer Sparrow',
           'Chipping Sparrow',
           'Clay-colored Sparrow',
           'House Sparrow',
           'Field Sparrow',
           'Fox Sparrow',
           'Grasshopper Sparrow',
           'Harris Sparrow',
           'Henslow Sparrow',
           'Le Conte Sparrow',
           'Lincoln Sparrow',
           'Nelson Sharp-tailed Sparrow',
           'Savannah Sparrow',
           'Seaside Sparrow',
           'Song Sparrow',
           'Tree Sparrow',
           'Vesper Sparrow',
           'White-crowned Sparrow',
           'White-throated Sparrow',
           'Cape Glossy Starling',
           'Bank Swallow',
           'Barn Swallow',
           'Cliff Swallow',
           'Tree Swallow',
           'Scarlet Tanager',
           'Summer Tanager',
           'Artic Tern',
           'Black Tern',
           'Caspian Tern',
           'Common Tern',
           'Elegant Tern',
           'Forsters Tern',
           'Least Tern',
           'Green-tailed Towhee',
           'Brown Thrasher',
           'Sage Thrasher',
           'Black-capped Vireo',
           'Blue-headed Vireo',
           'Philadelphia Vireo',
           'Red-eyed Vireo',
           'Warbling Vireo',
           'White-eyed Vireo',
           'Yellow-throated Vireo',
           'Bay-breasted Warbler',
           'Black-and-white Warbler',
           'Black-throated Blue Warbler',
           'Blue-winged Warbler',
           'Canada Warbler',
           'Cape May Warbler',
           'Cerulean Warbler',
           'Chestnut-sided Warbler',
           'Golden-winged Warbler',
           'Hooded Warbler',
           'Kentucky Warbler',
           'Magnolia Warbler',
           'Mourning Warbler',
           'Myrtle Warbler',
           'Nashville Warbler',
           'Orange-crowned Warbler',
           'Palm Warbler',
           'Pine Warbler',
           'Prairie Warbler',
           'Prothonotary Warbler',
           'Swainson Warbler',
           'Tennessee Warbler',
           'Wilson Warbler',
           'Worm-eating Warbler',
           'Yellow Warbler',
           'Northern Waterthrush',
           'Louisiana Waterthrush',
           'Bohemian Waxwing',
           'Cedar Waxwing',
           'American Three-toed Woodpecker',
           'Pileated Woodpecker',
           'Red-bellied Woodpecker',
           'Red-cockaded Woodpecker',
           'Red-headed Woodpecker',
           'Downy Woodpecker',
           'Bewick Wren',
           'Cactus Wren',
           'Carolina Wren',
           'House Wren',
           'Marsh Wren',
           'Rock Wren',
           'Winter Wren',
           'Common Yellowthroat']


def register_dataset(root):
    ds_name = 'cub_200'
    root = os.path.join(root, "CUB_200_2011")

    for split, image_dirname, sem_seg_dirname, class_names in [
        ("test", "images_detectron2/test", "annotations_detectron2/test", CLASSES),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        full_name = f"{ds_name}_sem_seg_{split}"
        DatasetCatalog.register(
            full_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(full_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_classes=class_names,
            stuff_colors=colormap(rgb=True),
            classes_of_interest=list(range(1, len(class_names))),
            background_class=0,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_dataset(_root)
