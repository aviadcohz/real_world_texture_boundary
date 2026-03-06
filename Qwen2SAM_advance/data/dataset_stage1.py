"""
Stage 1 dataset: COCO-format datasets for general segmentation recovery.

Wraps SAM3's existing Sam3ImageDataset and collator pipeline.
The only addition is extracting raw PIL images for Qwen's vision encoder.

SAM3's collator already preserves raw_images in BatchedDatapoint,
so we just need to configure the dataset correctly.
"""

import sys
sys.path.insert(0, "/home/aviad/sam3")

from typing import Optional

from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
from sam3.train.data.collator import collate_fn_api
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    PadToSizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.train.transforms.filter_query_transforms import (
    FilterCrowds,
    FilterEmptyTargets,
    FlexibleFilterFindGetQueries,
)
from sam3.train.transforms.segmentation import DecodeRle


def build_coco_transforms(
    resolution: int = 1008,
    min_scale: float = 0.5,
    max_scale: float = 1.5,
    with_masks: bool = True,
):
    """Build SAM3-style transforms for COCO training."""
    transforms = [
        FlexibleFilterFindGetQueries(FilterCrowds()),
    ]
    if with_masks:
        transforms.append(DecodeRle())
    transforms.extend([
        RandomResizeAPI(
            sizes=list(range(int(resolution * min_scale), int(resolution * max_scale), 32)),
            consistent_transform=True,
            max_size=resolution,
        ),
        PadToSizeAPI(size=resolution, consistent_transform=True),
        ToTensorAPI(),
        FlexibleFilterFindGetQueries(FilterEmptyTargets()),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        FlexibleFilterFindGetQueries(FilterEmptyTargets()),
    ])
    return ComposeAPI(transforms)


def build_coco_dataset(
    annotation_file: str,
    image_dir: str,
    resolution: int = 1008,
    with_masks: bool = True,
    prompts: Optional[str] = None,
    max_annotations_per_image: int = 200,
):
    """
    Build a COCO-format dataset for Stage 1 training.

    Args:
        annotation_file: Path to COCO JSON annotations
            (e.g., /data/coco/annotations/instances_train2017.json)
        image_dir: Path to image directory
            (e.g., /data/coco/train2017)
        resolution: Input resolution (1008 for SAM3)
        with_masks: Whether to load segmentation masks
        prompts: Optional custom prompts for categories
        max_annotations_per_image: Max annotations per image

    Returns:
        Sam3ImageDataset instance
    """
    from functools import partial

    coco_loader = partial(
        COCO_FROM_JSON,
        prompts=prompts,
        include_negatives=True,
    )

    transforms = build_coco_transforms(
        resolution=resolution,
        with_masks=with_masks,
    )

    dataset = Sam3ImageDataset(
        img_folder=image_dir,
        ann_file=annotation_file,
        coco_json_loader=coco_loader,
        transforms=transforms,
        max_ann_per_img=max_annotations_per_image,
        multiplier=1,
        training=True,
        load_segmentation=with_masks,
    )

    return dataset


def build_collate_fn(with_masks: bool = True):
    """Build SAM3's collator function."""
    from functools import partial
    return partial(
        collate_fn_api,
        repeats=0,
        dict_key="all",
        with_seg_masks=with_masks,
    )
