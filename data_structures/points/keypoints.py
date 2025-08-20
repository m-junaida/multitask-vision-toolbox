from abc import ABCMeta
from typing import Union, Sequence
import numpy as np
import torch
from torch import Tensor


IndexType = Union[slice, int, list, torch.LongTensor, torch.cuda.LongTensor,
                  torch.BoolTensor, torch.cuda.BoolTensor, np.ndarray]

class Keypoints(metaclass=ABCMeta):
    """Data structure for storing points of shape (N, K, 3).
    Supports both visibility or confidence scores as 3rd value.

    Args:
        data (np.ndarray or torch.Tensor): (N, K, 3) coordinates.
            N is number of instances, K is number of points, 3 is (x, y, score).
    """

    def __init__(self, data: Union[Tensor, np.ndarray, Sequence]):
        
        if isinstance(data, (np.ndarray, Tensor, Sequence)):
            data = torch.as_tensor(data)
        else:
            raise TypeError('points should be Tensor, ndarray, or Sequence, ',
                            f'but got {type(data)}')

        if data.numel() == 0:
            data = torch.empty((0, 0, 3), dtype=torch.float32)

        assert data.ndim == 3 and data.shape[2] == 3, "Invalid keypoint shape"
        self.tensor = data # (N, K, 3)

    def flip(self, img_shape, direction='horizontal'):
        """Flip points horizontally or vertically."""
        img_height, img_width = img_shape[:2]
        
        if direction == 'horizontal':
            self.tensor[..., 0] = img_width - self.tensor[..., 0]
        elif direction == 'vertical':
            self.tensor[..., 1] = img_height - self.tensor[..., 1]
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'.")

    def scale(self, scale_x, scale_y):
        self.tensor[..., 0] *= scale_x
        self.tensor[..., 1] *= scale_y

    def shift(self, x0, y0):
        self.tensor[..., 0] -= x0
        self.tensor[..., 1] -= y0

    def rotate(self, angle, center_x=0, center_y=0):
        """Rotate keypoints around a center point."""
        raise NotImplementedError("Rotation not implemented for KeypointsData.")

    def __len__(self):
        return self.tensor.size(0)

    def __deepcopy__(self, memo):
        """Only clone the ``self.tensor`` when applying deepcopy."""
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other
        other.tensor = self.tensor.clone()
        return other

    def to(self, *args, **kwargs):
        """Reload ``to`` from self.tensor."""
        return type(self)(self.tensor.to(*args, **kwargs), clone=False)

    def __repr__(self) -> str:
        """Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n' + str(self.tensor) + ')'