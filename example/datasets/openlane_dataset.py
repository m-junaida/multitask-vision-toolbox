from typing import Tuple, List, Dict
import numpy as np
from imagesize import get as get_image_size
from mmengine.registry import DATASETS
from utils.misc import read_json, to_one_hot
from datasets.multitask_dataset import MultiTaskDataset
import pandas as pd
from pathlib import Path
from data_structures import MultiLabelData

@DATASETS.register_module()
class OpenLaneDataset(MultiTaskDataset):
    def __init__(
        self,
        ann_prefix: str,
        img_prefix: str,
        pipeline,
        image_set_file: str,
        task_classes_map: Dict[str, Dict[str, int]]=None,
        data_root=None,
        metainfo=None,
        test_mode=False,
        filter_cfg=dict(),
        backend_args:dict = None,
        use_cache=True,
        **kwargs
    ):
        self.scene_annotations = pd.read_json(Path(ann_prefix) / "scene.json").T
        
        super(OpenLaneDataset, self).__init__(
            ann_prefix=ann_prefix,
            img_prefix=img_prefix,
            pipeline=pipeline,
            image_set_file=image_set_file,
            task_classes_map=task_classes_map,
            data_root=data_root,
            metainfo=metainfo,
            test_mode=test_mode,
            filter_cfg=filter_cfg,
            backend_args=backend_args,
            use_cache=use_cache,
            **kwargs
        )
        
    
    def get_lane_kpts(self, annotations: dict) -> np.ndarray:
        """
        Get lane keypoints from the given annotations dict.
        Args:
            annotations (dict): The annotations dict containing lane information.
        Returns:
            np.ndarray: An array of lane keypoints.
        """
        max_kpts = max([np.array(lane['uv']).shape[1] for lane in annotations['lane_lines'] if lane['attribute'] > 0] + [0])
        lanes_kpts = np.zeros((4, max_kpts, 3), dtype=np.float32)
        for lane in annotations['lane_lines']:
            if lane['attribute'] > 0: # only considering most important 4 lanes here
                lane_kpts = np.concatenate([
                    np.array(lane['uv']),
                    np.ones_like(lane['uv'][0]).reshape(1, -1)
                ], axis=0)
                lanes_kpts[lane['attribute'] - 1, :lane_kpts.shape[1]] = lane_kpts.T
        
        return lanes_kpts

    def get_classification_labels(self, annotations: dict) -> np.ndarray:
        labels = []
        for key, value in annotations.items():
            labels.append(self.task_classes_map[f"classification"][value])

        return labels
    
    def get_instance_annotations(self, annotations: dict) -> List[dict]:
        """
        Get instance annotations from the given annotations dict.
        See parent class method for more details.
        """
        bboxes = []
        for bbox in annotations['result']:
            bboxes.append(dict(
                bbox=[bbox['x'], bbox['y'], bbox['x']+bbox['width'], bbox['y']+bbox['height']],
                bbox_label=bbox['type'], # Already and id in int, no need to map
                ignore_flag=False
            ))

        return bboxes

    def get_annotations(self, image_name: str) -> dict:
        cipo_annotations = read_json((self.ann_prefix / "cipo" / (image_name + ".json")))
        lane_annotations = read_json((self.ann_prefix / "lane_lines" / image_name).with_suffix('.json'))
        scene_annotations = self.scene_annotations.loc[Path(image_name).parent.name].to_dict()
        # og_img
        w,h = get_image_size(self.img_prefix / image_name)

        return dict(
            img_path=(self.img_prefix / image_name),
            width=w,
            height=h,
            instances=self.get_instance_annotations(cipo_annotations),
            lanes_kpts=self.get_lane_kpts(lane_annotations),
            classification_labels=self.get_classification_labels(scene_annotations)
        )
