
# Copyright (c) Facebook, Inc. and its affiliates.
import os

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_sem_seg
    from detectron2.utils.colormap import colormap
except:
    from mess.utils.catalog import DatasetCatalog, MetadataCatalog
    from mess.utils.data import load_sem_seg
    from mess.utils.colormap import colormap

# Lists from CAT-Seg (OVSeg uses another ordering)
PASCALCONTEX59_NAMES = ["aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird", "boat", "book", "bottle", "building", "bus", "cabinet", "car", "cat", "ceiling", "chair", "cloth", "computer", "cow", "cup", "curtain", "dog", "door", "fence", "floor", "flower", "food", "grass", "ground", "horse", "keyboard", "light", "motorbike", "mountain", "mouse", "person", "plate", "platform", "pottedplant", "road", "rock", "sheep", "shelves", "sidewalk", "sign", "sky", "snow", "sofa", "diningtable", "track", "train", "tree", "truck", "tvmonitor", "wall", "water", "window", "wood"]

PASCALCONTEX459_NAMES = ["accordion", "aeroplane", "airconditioner", "antenna", "artillery", "ashtray", "atrium", "babycarriage", "bag", "ball", "balloon", "bambooweaving", "barrel", "baseballbat", "basket", "basketballbackboard", "bathtub", "bed", "bedclothes", "beer", "bell", "bench", "bicycle", "binoculars", "bird", "birdcage", "birdfeeder", "birdnest", "blackboard", "board", "boat", "bone", "book", "bottle", "bottleopener", "bowl", "box", "bracelet", "brick", "bridge", "broom", "brush", "bucket", "building", "bus", "cabinet", "cabinetdoor", "cage", "cake", "calculator", "calendar", "camel", "camera", "cameralens", "can", "candle", "candleholder", "cap", "car", "card", "cart", "case", "casetterecorder", "cashregister", "cat", "cd", "cdplayer", "ceiling", "cellphone", "cello", "chain", "chair", "chessboard", "chicken", "chopstick", "clip", "clippers", "clock", "closet", "cloth", "clothestree", "coffee", "coffeemachine", "comb", "computer", "concrete", "cone", "container", "controlbooth", "controller", "cooker", "copyingmachine", "coral", "cork", "corkscrew", "counter", "court", "cow", "crabstick", "crane", "crate", "cross", "crutch", "cup", "curtain", "cushion", "cuttingboard", "dais", "disc", "disccase", "dishwasher", "dock", "dog", "dolphin", "door", "drainer", "dray", "drinkdispenser", "drinkingmachine", "drop", "drug", "drum", "drumkit", "duck", "dumbbell", "earphone", "earrings", "egg", "electricfan", "electriciron", "electricpot", "electricsaw", "electronickeyboard", "engine", "envelope", "equipment", "escalator", "exhibitionbooth", "extinguisher", "eyeglass", "fan", "faucet", "faxmachine", "fence", "ferriswheel", "fireextinguisher", "firehydrant", "fireplace", "fish", "fishtank", "fishbowl", "fishingnet", "fishingpole", "flag", "flagstaff", "flame", "flashlight", "floor", "flower", "fly", "foam", "food", "footbridge", "forceps", "fork", "forklift", "fountain", "fox", "frame", "fridge", "frog", "fruit", "funnel", "furnace", "gamecontroller", "gamemachine", "gascylinder", "gashood", "gasstove", "giftbox", "glass", "glassmarble", "globe", "glove", "goal", "grandstand", "grass", "gravestone", "ground", "guardrail", "guitar", "gun", "hammer", "handcart", "handle", "handrail", "hanger", "harddiskdrive", "hat", "hay", "headphone", "heater", "helicopter", "helmet", "holder", "hook", "horse", "horse-drawncarriage", "hot-airballoon", "hydrovalve", "ice", "inflatorpump", "ipod", "iron", "ironingboard", "jar", "kart", "kettle", "key", "keyboard", "kitchenrange", "kite", "knife", "knifeblock", "ladder", "laddertruck", "ladle", "laptop", "leaves", "lid", "lifebuoy", "light", "lightbulb", "lighter", "line", "lion", "lobster", "lock", "machine", "mailbox", "mannequin", "map", "mask", "mat", "matchbook", "mattress", "menu", "metal", "meterbox", "microphone", "microwave", "mirror", "missile", "model", "money", "monkey", "mop", "motorbike", "mountain", "mouse", "mousepad", "musicalinstrument", "napkin", "net", "newspaper", "oar", "ornament", "outlet", "oven", "oxygenbottle", "pack", "pan", "paper", "paperbox", "papercutter", "parachute", "parasol", "parterre", "patio", "pelage", "pen", "pencontainer", "pencil", "person", "photo", "piano", "picture", "pig", "pillar", "pillow", "pipe", "pitcher", "plant", "plastic", "plate", "platform", "player", "playground", "pliers", "plume", "poker", "pokerchip", "pole", "pooltable", "postcard", "poster", "pot", "pottedplant", "printer", "projector", "pumpkin", "rabbit", "racket", "radiator", "radio", "rail", "rake", "ramp", "rangehood", "receiver", "recorder", "recreationalmachines", "remotecontrol", "road", "robot", "rock", "rocket", "rockinghorse", "rope", "rug", "ruler", "runway", "saddle", "sand", "saw", "scale", "scanner", "scissors", "scoop", "screen", "screwdriver", "sculpture", "scythe", "sewer", "sewingmachine", "shed", "sheep", "shell", "shelves", "shoe", "shoppingcart", "shovel", "sidecar", "sidewalk", "sign", "signallight", "sink", "skateboard", "ski", "sky", "sled", "slippers", "smoke", "snail", "snake", "snow", "snowmobiles", "sofa", "spanner", "spatula", "speaker", "speedbump", "spicecontainer", "spoon", "sprayer", "squirrel", "stage", "stair", "stapler", "stick", "stickynote", "stone", "stool", "stove", "straw", "stretcher", "sun", "sunglass", "sunshade", "surveillancecamera", "swan", "sweeper", "swimring", "swimmingpool", "swing", "switch", "table", "tableware", "tank", "tap", "tape", "tarp", "telephone", "telephonebooth", "tent", "tire", "toaster", "toilet", "tong", "tool", "toothbrush", "towel", "toy", "toycar", "track", "train", "trampoline", "trashbin", "tray", "tree", "tricycle", "tripod", "trophy", "truck", "tube", "turtle", "tvmonitor", "tweezers", "typewriter", "umbrella", "unknown", "vacuumcleaner", "vendingmachine", "videocamera", "videogameconsole", "videoplayer", "videotape", "violin", "wakeboard", "wall", "wallet", "wardrobe", "washingmachine", "watch", "water", "waterdispenser", "waterpipe", "waterskateboard", "watermelon", "whale", "wharf", "wheel", "wheelchair", "window", "windowblinds", "wineglass", "wire", "wood", "wool"]


def _get_voc_meta(cat_list):
    ret = {
        "stuff_classes": cat_list,
    }
    return ret


def register_pascal_context_59(root):
    root = os.path.join(root, "VOCdevkit/VOC2010")
    meta = _get_voc_meta(PASCALCONTEX59_NAMES)
    for name, image_dirname, sem_seg_dirname in [
        ("val", "JPEGImages", "annotations_detectron2/pc59_val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"pascal_context_59_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_colors=colormap(rgb=True),
            **meta,
        )

def register_pascal_context_459(root):
    root = os.path.join(root, "VOCdevkit/VOC2010")
    meta = _get_voc_meta(PASCALCONTEX459_NAMES)
    for name, image_dirname, sem_seg_dirname in [
        ("val", "JPEGImages", "annotations_detectron2/pc459_val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"pascal_context_459_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="tif", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
            stuff_colors=colormap(rgb=True),
            background_class=65535,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "")
if 'pascal_context_59_sem_seg_val' not in DatasetCatalog.list():
    register_pascal_context_59(_root)
else:
    # Make sure the class order in the annotations is matching the registered classes
    pc_59_classes = MetadataCatalog.get('pascal_context_59_sem_seg_val').stuff_classes
    for annotated, registered in zip(PASCALCONTEX59_NAMES, pc_59_classes):
        assert annotated == registered, f"pascal context 59 class name mismatch: {annotated} vs {registered}"

    MetadataCatalog.get('pascal_context_59_sem_seg_val').set(
        stuff_colors=colormap(rgb=True),
    )

if 'pascal_context_459_sem_seg_val' not in DatasetCatalog.list():
    register_pascal_context_459(_root)
else:
    # Make sure the class order in the annotations is matching the registered classes
    pc_459_classes = MetadataCatalog.get('pascal_context_459_sem_seg_val').stuff_classes
    for annotated, registered in zip(PASCALCONTEX459_NAMES, pc_459_classes):
        assert annotated == registered, f"pascal context 459 class name mismatch: {annotated} vs {registered}"

    MetadataCatalog.get('pascal_context_459_sem_seg_val').set(
        stuff_colors=colormap(rgb=True),
    )