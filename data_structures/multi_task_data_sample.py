from mmdet.structures import DetDataSample
from mmengine.structures import PixelData, InstanceData
from data_structures import MultiLabelData

class MultiTaskDataSample(DetDataSample):
    """Data structure for multi-task learning in CV.

    Extends `DetDataSample` to support multiple CV tasks like detection,
    segmentation, keypoints, lane detection, depth estimation, etc.

    Inherits:
        - `DetDataSample.gt_instances` for ground-truth bboxes, labels, masks, etc.
        - `DetDataSample.pred_instances` for predicted versions
        - `DetDataSample.gt_sem_seg` for semantic segmentation
        - `DetDataSample.metainfo` and `DetDataSample.data_fields` for metadata

    Additional fields:
        - `gt_keypoints` (InstanceData): Ground truth keypoints
        - `pred_keypoints` (InstanceData): Predicted keypoints
        - `gt_lanes` (InstanceData): Ground truth lane lines
        - `pred_lanes` (InstanceData): Predicted lane lines
        - `gt_depth` (PixelData): Ground truth depth maps
        - `pred_depth` (PixelData): Predicted depth maps
        - `gt_flow` (PixelData): Ground truth optical flow
        - `pred_flow` (PixelData): Predicted optical flow
        - `gt_cls_label` (LabelData): For classification tags
        - `pred_cls_label` (LabelData): For classification predictions

    You can access and assign each field like:
        - `sample.gt_keypoints = InstanceData(...)`
        - `sample.pred_depth = PixelData(...)`
        - `sample.gt_cls_label = LabelData(...)`
    """

    @property
    def gt_keypoints(self) -> InstanceData:
        return self.get('gt_keypoints', None)

    @gt_keypoints.setter
    def gt_keypoints(self, value: InstanceData):
        self.set_field(value, 'gt_keypoints', dtype=InstanceData)

    @property
    def pred_keypoints(self) -> InstanceData:
        return self.get('pred_keypoints', None)

    @pred_keypoints.setter
    def pred_keypoints(self, value: InstanceData):
        self.set_field(value, 'pred_keypoints', dtype=InstanceData)

    @property
    def gt_lanes(self) -> InstanceData:
        return self.get('gt_lanes', None)

    @gt_lanes.setter
    def gt_lanes(self, value: InstanceData):
        self.set_field(value, 'gt_lanes', dtype=InstanceData)

    @property
    def pred_lanes(self) -> InstanceData:
        return self.get('pred_lanes', None)

    @pred_lanes.setter
    def pred_lanes(self, value: InstanceData):
        self.set_field(value, 'pred_lanes', dtype=InstanceData)

    @property
    def gt_depth(self) -> PixelData:
        return self.get('gt_depth', None)

    @gt_depth.setter
    def gt_depth(self, value: PixelData):
        self.set_field(value, 'gt_depth', dtype=PixelData)

    @property
    def pred_depth(self) -> PixelData:
        return self.get('pred_depth', None)

    @pred_depth.setter
    def pred_depth(self, value: PixelData):
        self.set_field(value, 'pred_depth', dtype=PixelData)

    @property
    def gt_flow(self) -> PixelData:
        return self.get('gt_flow', None)

    @gt_flow.setter
    def gt_flow(self, value: PixelData):
        self.set_field(value, 'gt_flow', dtype=PixelData)

    @property
    def pred_flow(self) -> PixelData:
        return self.get('pred_flow', None)

    @pred_flow.setter
    def pred_flow(self, value: PixelData):
        self.set_field(value, 'pred_flow', dtype=PixelData)

    @property
    def gt_classification_label(self) -> MultiLabelData:
        return self.get('gt_classification_label', None)

    @gt_classification_label.setter
    def gt_classification_label(self, value: MultiLabelData):
        self.set_field(value, 'gt_classification_label', dtype=MultiLabelData)

    @property
    def pred_classification_label(self) -> MultiLabelData:
        return self.get('pred_classification_label', None)

    @pred_classification_label.setter
    def pred_classification_label(self, value: MultiLabelData):
        self.set_field(value, 'pred_classification_label', dtype=MultiLabelData)
