from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs
from mmcv.transforms import to_tensor
from mmengine.structures import PixelData, InstanceData
from data_structures import MultiLabelData

@TRANSFORMS.register_module()
class PackMultiTaskInputs(PackDetInputs):
    
    def transform(self, results: dict) -> dict:
        
        packed_results = super().transform(results)
        # The parent class PackDetInputs packs the data samples with following fields:
        # - inputs: tensor of input image
        # - data_samples: DetDataSample object containing metadata and annotations with following fields like:
        #     - gt_instances: InstanceData for ground truth bboxes, masks, keypoints, etc.
        #     - proposals: InstanceData for proposals
        #     - gt_sem_seg: PixelData for semantic segmentation
        #     - metainfo: metadata about the image
    
        # we are packing our custom fields
        if "gt_classification_labels" in results:
            packed_results['data_samples'].gt_classification_labels = MultiLabelData(labels=to_tensor(results["gt_classification_labels"]))
        if "gt_keypoints" in results:
            packed_results['data_samples'].instances.keypoints = results["gt_keypoints"]
        if "gt_lane_kpts"  in results:
            packed_results['data_samples'].gt_lane_lines = results["gt_lane_kpts"]
        if "gt_depths" in results:
            packed_results['data_samples'].gt_depth = PixelData(depth=to_tensor(results['gt_depth']))
        ## We can add more fields as needed

        return packed_results