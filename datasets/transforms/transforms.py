from typing import Tuple, Union
import numpy as np
import mmcv
from mmcv.transforms.utils import cache_randomness
from mmengine.registry import TRANSFORMS
from mmdet.datasets.transforms.transforms import RandomCrop, Resize, RandomFlip

@TRANSFORMS.register_module()
class MultiTaskResize(Resize):
    """
    MMDet Resize function resizes Image, Bounding Boxes and Segmentation map.
    wanted to resize other things such as keypoints
    """

    def _resize_lane_kpts(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_lane_kpts', None) is not None:
            keypoints = results['gt_lane_kpts']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['img_shape'][0])
            results['gt_lane_kpts'] = keypoints

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        self._resize_keypoints(results)
        return results
    

@TRANSFORMS.register_module()
class MultiTaskRandomFlip(RandomFlip):
    """MMDet Flip function flips Image, Bounding Boxes and Segmentation map.
    wanted to flip other things such as keypoints
    """

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, semantic segmentation map, keypoints, lane_line_keypoints, depth map."""
        super()._flip(results)
        img_shape = results['img'].shape[:2]
        
        if 'gt_keypoints' in results:
            results['gt_keypoints'] = results['gt_keypoints'].flip(
                img_shape, results['flip_direction'])
        
        if 'gt_lane_kpts' in results:
            results['gt_lane_kpts'] = results['gt_lane_kpts'].flip(
                img_shape, results['flip_direction'])
        
        if 'gt_depth_map' in results:
            results['gt_depth_map'] = mmcv.imflip(
                results['gt_depth_map'], direction=results['flip_direction'])


@TRANSFORMS.register_module()
class FixedCrop(RandomCrop):
    """Fix crop the image & bboxes & masks. based on `crop_size`

    It will start from the crop_tl (x,y) corner of the image and will go upto crop_br (x,y) corner

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)
    - gt_seg_map (optional)
    - gt_instances_ids (options, only used in MOT/VIS)

    Added Keys:

    - homography_matrix

    Args:
        crop_tl (tuple): The absolute starting point (x_min, y_min) of the crop. top left corner of the cropped image
        crop_br (tuple): The absolute ending point (x_max, y_max) of the crop. bottom right corner of the cropped image
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Defaults to False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Defaults to False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          ``gt_bboxes`` corresponds to ``gt_labels`` and ``gt_masks``, and
          ``gt_bboxes_ignore`` corresponds to ``gt_labels_ignore`` and
          ``gt_masks_ignore``.
        - If the crop does not contain any gt-bbox region and
          ``allow_negative_crop`` is set to False, skip this image.
    """

    def __init__(self,
                 crop_tl: tuple, 
                 crop_br: tuple,
                 allow_negative_crop: bool = False,
                 recompute_bbox: bool = False,
                 bbox_clip_border: bool = True) -> None:
        
        # both are (x,y)
        self.crop_tl = crop_tl
        self.crop_br = crop_br
        # computing crop_size for parent class (widht, height)
        # (x_max - x_min, y_max - y_min) -> (width, height)
        crop_size = (crop_br[0] - crop_tl[0], crop_br[1] - crop_tl[1])
        super(FixedCrop, self).__init__(crop_size, 'absolute', allow_negative_crop, recompute_bbox, bbox_clip_border)


    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        """Overriding the paraent class function
        In this we will not be making random offset. 
        We will return the crop_tl

        Args:
            margin (Tuple[int, int]): will not be used here

        Returns:
            Tuple[int, int]: fixed crop_tl will be returned.
        """

        # Original function return offset_h, offset_w
        return self.crop_tl[1], self.crop_tl[0]

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:
        results = super(FixedCrop, self)._crop_data(results, crop_size, allow_negative_crop)
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'][:, :, :2] = results['gt_keypoints'][:, :, :2] - self.crop_tl
        return results

    def transform(self, results: dict) -> Union[dict, None]:
        results['crop_tl'] = self.crop_tl
        results['crop_br'] = self.crop_br
        return super(FixedCrop, self).transform(results)
