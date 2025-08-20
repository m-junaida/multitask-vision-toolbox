from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.structures.mask import BitmapMasks
from data_structures import MultiLabelData, Keypoints, LaneLinesKeypoints

from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadMultiTaskAnnotations(LoadAnnotations):
    """
    Custom loader for multi-task annotations (e.g., detection, classification, keypoints, lanes).
    Extends LoadAnnotations to support additional fields.

        The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of enclosed object.
                'bbox_label': 1,

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

                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
            # List of lane lines, each line is a list of points in (x, y, v) order
            'lane_lines': list[list[float]],  # Optional, for lane lines
            # List of classification labels, each label is a string
            'classification_labels': list[str],  # Optional, for classification labels
            # Depth map, if available
            'depth': list[list[float]],  # Optional, for depth maps
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': BaseBoxes(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
            'gt_keypoints': np.ndarray (N, K, 3)
             # In (x, y, v) order, float type.
            'gt_lane_lines': np.ndarray (L, K, 3)
             # In int type.
            'gt_classification_labels': np.ndarray (N, )
        }

    Required Keys:

    - height
    - width
    - instances

      - bbox (optional)
      - bbox_label
      - mask (optional)
      - ignore_flag
      - keypoints (optional)

    - seg_map_path (optional)
    - lanes_kpts (optional)
    - classification_labels (optional)
    - depth (optional)

    Added Keys:

    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (bool)
    - gt_keypoints (np.ndarray) TODO: (KeypointsData)
    - gt_lanes_kpts (np.ndarray) TODO: wrap in (LaneLinesData)
    - gt_classification_labels (np.ndarray) TODO: wrap in (MultiLabelData)
    """
    def __init__(self, with_bbox=True, with_bbox_label=True, with_mask=False, with_seg=False,
                 with_keypoints=False, with_lane_kpts=False, with_classification_labels=False, 
                 with_depth=False, **kwargs):
        # using parent class to initialize common fields
        super().__init__(with_bbox=with_bbox, with_label=with_bbox_label,
                         with_mask=with_mask, with_seg=with_seg, **kwargs)
        # parent class will 
        self.with_keypoints = with_keypoints
        self.with_lane_kpts = with_lane_kpts
        self.with_classification_labels = with_classification_labels
        self.with_depth = with_depth

    def load_depth(self, results):
        ## TODO: load depth
        return None

    def transform(self, results):
        results = super().transform(results)
        # parent class will give following keys in results:
        # gt_bboxes: np.ndarray or BaseBoxes
        # gt_ignore_flags: np.ndarray
        # gt_bboxes_labels: np.ndarray
        # gt_masks: BitmapMasks or PolygonMasks
        # gt_seg_map: np.ndarray    
        # ignore_index: int

        if self.with_keypoints:
            results['gt_keypoints'] = Keypoints(self._load_kps(results))
        
        if self.with_lane_kpts:
            results['gt_lanes_kpts'] = LaneLinesKeypoints(results['lanes_kpts'])

        if self.with_classification_labels:
            results['gt_classification_labels'] = MultiLabelData(results['classification_labels'])
        
        if self.with_depth:
            results['gt_depth'] = self.load_depth(results)
        # We can add more fields as needed
        # TODO: Will create basic structure for common datatypes and wrap them in appropriate classes

        return results
