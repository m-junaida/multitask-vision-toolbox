from typing import List, Dict
from abc import abstractmethod
import mmengine
import numpy as np
from mmengine.registry import DATASETS
from mmengine.dataset import BaseDataset
import os
from pathlib import Path
import pickle as pkl

@DATASETS.register_module()
class MultiTaskDataset(BaseDataset):
    def __init__(
        self,
        ann_prefix: str,
        img_prefix: str,
        pipeline,
        image_set_file: str,
        task_classes_map: Dict[str, Dict[str, int]],
        data_root=None,
        metainfo=None,
        test_mode=False,
        filter_cfg=dict(),
        backend_args:dict = None,
        use_cache=True,
        **kwargs
    ):
        """
        task_classes_map: Dict[str, Dict[str, int]]
            A mapping of task name to class names and their corresponding class ids.
            For example: These are just random examples
            {
                'bbox': {'car': 0, 'bus': 1, 'person': 2},
                'classification': {'daytime': 0, 'cloudy': 1},
                'keypoints': {'nose': 0, 'left_eye': 1, 'right_eye': 2},
                'lane_kpts': {'left_lane': 0, 'right_lane': 1}
            }

        metainfo: meta information dict
            It is a dict containing the class_names in order. key is task name. 
            For example:
            {
                'bbox_classes': ['car', 'bus', 'person'] For Detection with class_ids = [0,1,2]
                'classification_classes': ['daytime', 'cloudy']  For Classification with class_ids = [0,1]
                'keypoints_classes': ['nose', 'left_eye', 'right_eye'] For Keypoints with class_ids = [0,1,2]
                'lane_kpts_classes': ['left_lane', 'right_lane'] For Lane Lines with class_ids = [0,1]
            }.
        """

        self.image_set_file = image_set_file
        self.ann_prefix = Path(ann_prefix)
        self.img_prefix = Path(img_prefix)
        self.task_classes_map = task_classes_map
        self.use_cache = use_cache

        self.img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]

        # calling the parent class's init
        # it will call full_init() in BaseDataset class
        # this function will call load_data_list()
        super(MultiTaskDataset, self).__init__(
            metainfo=metainfo,
            data_root=data_root,
            filter_cfg=filter_cfg,
            serialize_data=False,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs
        )

    def load_data_list(self) -> List[dict]:
        if self.image_set_file is not None:
            images_name_list = mmengine.list_from_file(self.image_set_file)
        else:
            # Recursively find all .json files under img_prefix, return relative paths
            images_name_list = [
                str(p.relative_to(self.img_prefix))
                for ext in self.img_extensions
                for p in self.img_prefix.rglob(ext)
            ]

        if self.use_cache and self.image_set_file and os.path.exists(self.image_set_file + ".pkl"):
            data_list = pkl.load(open(self.image_set_file + ".pkl", "rb"))
        else:
            data_list = []
            for image_name in images_name_list:
                data_list.append(self.get_annotations(image_name))
            #data_list = p_map(self.get_annotations, ann_list)
            data_list = [di for di in data_list if di is not None]

            if self.use_cache and self.image_set_file:
                pkl.dump(data_list, open(self.image_set_file + ".pkl", "wb"))

        print(f"{len(data_list)}/{len(images_name_list)} examples being used from {self.image_set_file if self.image_set_file is not None else str(self.ann_prefix)}")
        return data_list

    def get_data_info(self, idx: int) -> dict:
        ann_info = dict(**self.data_list[idx])
        ann_info["img_id"] = idx
        return ann_info
        
    @abstractmethod
    def get_annotations(self, image_name: str) -> dict:
        """
        Get annotations for a given image name.
        Args:
            image_name (str): The name of the image file.
        Returns:
            dict: A dictionary containing the annotations for the image.

        User should implement this method to return the annotations in the required format.
        The expected format is:
        {
            'img_path': Path to the image file,
            'width': Width of the image,
            'height': Height of the image,
            'instances': List[dict]: A list of instance annotations. (optional)
                [
                    {
                        'bbox': [x1, y1, x2, y2],
                        'bbox_label': int,
                        'ignore_flag': bool,
                        
                        # Used in instance/panoptic segmentation. The segmentation mask
                        # of the instance or the information of segments.
                        # 1. If list[list[float]], it represents a list of polygons,
                        # one for each connected component of the object. Each
                        # list[float] is one simple polygon in the format of
                        # [x1, y1, ..., xn, yn] (n >= 3). The Xs and Ys are absolute
                        # coordinates in unit of pixels.
                        # 2. If dict, it represents the per-pixel segmentation mask in
                        # COCO's compressed RLE format. The dict should have keys
                        # “size” and “counts”.  Can be loaded by pycocotools
                        'mask': list[list[float]] or dict,
                        
                        # Used in key point detection.
                        # Can only load the format of [x1, y1, v1,…, xn, yn, vn]. v[i]
                        # means the visibility of this keypoint. n must be equal to the
                        # number of keypoint categories.
                        'keypoints': [x1, y1, v1, ..., xn, yn, vn]
                    },
                    ...
                ]
            'lane_kpts': Lane keypoints (L, K, 3) (optional),
            'classification_labels': Classification labels (N,) (optional),
            'seg_map_path': Path to the segmentation map file (optional), will be loaded in loading.py
            'depth_map_path': Path to the depth map file (optional), will be loaded in loading.py
        }
        """
        raise NotImplementedError