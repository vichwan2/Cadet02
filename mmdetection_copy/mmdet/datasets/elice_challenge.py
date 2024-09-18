# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset

from .coco import CocoDataset

@DATASETS.register_module()
class eliceAI_detection(CocoDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('crack', 'pothole' ),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32)]
    }