import torch
from data_structures import Keypoints

class LaneLinesKeypoints(Keypoints):
    """Data structure for storing lane_lines of shape (L, P, 3).
    Supports both visibility or confidence scores as 3rd value.

    Args:
        data (np.ndarray or torch.Tensor): (L, P, 3) coordinates.
                - L is number of lane lines
                - P is number of points per lane line,
                - 3 is (x, y, score).
            Lanes are represented as sequences of points. 
            This sequence is mostly from left to right, but can vary based on dataset.
            For Example, in KITTI, lanes are ordered from leftmost to rightmost.
    """

    def flip(self, img_width, img_height, direction='horizontal'):
        """Flip lane_lines horizontally or vertically.
        Args:
            img_width (int): Width of the image.
            img_height (int): Height of the image.
            direction (str): 'horizontal' or 'vertical'.
        
        Flipping horizontally means reversing the x-coordinates and also reversing the order of lines.
        the leftmost lane becomes the rightmost lane and vice versa.
        Flipping vertically means reversing the y-coordinates.
        """
        if direction == 'horizontal':
            self.tensor[..., 0] = img_width - self.tensor[..., 0] # reverse x-coordinates
            # reverse the order of lanes (left becomes right and vice versa)
            self.tensor = torch.flip(self.tensor, dims=[0]) # flip along the first dimension
        elif direction == 'vertical':
            self.tensor[..., 1] = img_height - self.tensor[..., 1]
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'.")
