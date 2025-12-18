from typing import Tuple, Optional, List
import torch
import triton
import triton.language as tl
from mmengine.structures import BaseDataElement
from mmengine.registry import MODELS
from models.heads.base_head import BaseHead

@triton.jit
def triton_interp_kernel(
    x_ptr, xp_ptr, fp_ptr, mask_ptr, out_ptr,
    query_min_ptr, query_max_ptr,
    NL, NP, NA,
    EPS: tl.constexpr,
    BLOCK_SIZE_NP: tl.constexpr,  # Block size for NP dimension
):
    """
    Interpolate points given a line. This is the main kernel which is 8x faster than naive implementation.

    Args:
        x_ptr: (NL, NA) tensor of query points
        xp_ptr: (NL, NP) tensor of points
        fp_ptr: (NL, NP) tensor of values corresponding to points
        mask_ptr: (NL, NP) tensor of mask/validity of points
        out_ptr: (NL, NA) tensor to store interpolated values
        query_min_ptr: (NL) tensor of minimum query points
        query_max_ptr: (NL) tensor of maximum query points
        NL: number of lines
        NP: number of points
        NA: number of anchors
    """
    # 2D indexing
    line_id = tl.program_id(0)
    anchor_id = tl.program_id(1)
    
    if line_id >= NL or anchor_id >= NA:
        return
    
    # Calculate indices
    x_idx = line_id * NA + anchor_id
    xp_start = line_id * NP
    
    # Load query point
    x_val = tl.load(x_ptr + x_idx)
    
    # Initialize as NaN
    result = float('nan')

    # Always load bounds (tensors always exist)
    q_min = tl.load(query_min_ptr + line_id)
    q_max = tl.load(query_max_ptr + line_id)
    # Check query bounds first (cheap)
    if (x_val < q_min) or (x_val > q_max):
        tl.store(out_ptr + x_idx, result)
        return
    
    offsets = tl.arange(0, BLOCK_SIZE_NP)
    mask = tl.load(mask_ptr + xp_start + offsets, mask=offsets < NP, other=0.0)
    # Need at least 2 valid points
    if tl.sum(mask) < 2:
        tl.store(out_ptr + x_idx, result)
        return
    
    # Binary search within valid range
    left = 0
    right = NP
    
    while left + 1 < right:
        mid = left + (right - left) // 2
        xp_mid = tl.load(xp_ptr + xp_start + mid)
        # Normal binary search
        if x_val < xp_mid:
            right = mid
        elif x_val > xp_mid:
            left = mid
        else:
            left = mid
            right = mid+1

    # Load interpolation values
    xp_left = tl.load(xp_ptr + xp_start + left)
    xp_right = tl.load(xp_ptr + xp_start + right)
    fp_left = tl.load(fp_ptr + xp_start + left)
    fp_right = tl.load(fp_ptr + xp_start + right)
    
    # Linear interpolation
    denominator = xp_right - xp_left
    if tl.abs(denominator) > EPS:
        alpha = (x_val - xp_left) / denominator
        alpha = tl.clamp(alpha, 0.0, 1.0)
        result = fp_left + alpha * (fp_right - fp_left)
    else:
        result = (fp_left + fp_right) / 2.0
    
    tl.store(out_ptr + x_idx, result)


def triton_interp(x, xp, fp, mask, query_min, query_max):
    """
    Optimized Triton implementation for similar to np.interp()
    """
    NL, NA = x.shape
    _, NP = xp.shape
    
    result = torch.full_like(x, float('nan'))
    
    # Make contiguous
    x = x.contiguous()
    xp = xp.contiguous()
    fp = fp.contiguous()
    mask = mask.contiguous()
    result = result.contiguous()
    
    # Choose block size
    BLOCK_SIZE_NP = min(128, triton.next_power_of_2(NP))
    
    grid = (NL, NA)
    # one processor per line and anchor
    triton_interp_kernel[grid](
        x, xp, fp, mask, result,
        query_min, query_max,
        NL, NP, NA,
        EPS=1e-6,
        BLOCK_SIZE_NP=BLOCK_SIZE_NP,
        num_warps=4
    )
    
    return result

@MODELS.register_module()
class UltraFastLaneDetectionV2(BaseHead):
    """
    Head for Lane Line Detection Task from Ultra-Fast-Lane-Detection-v2 (https://arxiv.org/pdf/2206.07389)
    Code replication from (https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/model/model_culane.py)

    Args:
        in_feature_size (tuple): Feature size of backbone/neck (whatever is inputed to the head)
            in_feature_size = (C, H, W)
        num_anchors_row (int): Number of anchors in row direction
        num_bins_row (int): Number of bins in row direction
        num_anchors_col (int): Number of anchors in col direction
        num_bins_col (int): Number of bins in col direction
        num_lanes_on_row (int): Number of lanes on row direction
        num_lanes_on_col (int): Number of lanes on col direction
        lanes_from_anchors_row (List[int]): Lane index from anchors in row direction
        lanes_from_anchors_col (List[int]): Lane index from anchors in col direction
        valid_thresh (float): Threshold for valid lane lines. Only lanes with more than valid_thresh points will be considered valid
        pooled_feat_ch (int): Dimension of pooled feature. [Default: 8 (original in paper)]
            the feature size after pooling will become (B, C, H, W) -> (B, pooled_feat_ch, H, W)
        mlp_mid_dim (int): Dimension of middle layer in fc layers. [Default: 2048 (original in paper)]
        fc_norm (bool): Whether to use batchnorm in fc layers
        feat_idx (int): Feature index of backbone/neck (it will be a Pyramid Feature) Default: -1 meaning last
        name (str): Name of the head
    """
    def __init__(
        self, 
        in_feature_size: Tuple[int, int, int],
        num_anchors_row: int, 
        num_bins_row: int, 
        num_lanes_on_row: int = 4, 
        num_anchors_col: int = 0, 
        num_bins_col: int = 0, 
        num_lanes_on_col: int = 0,
        lanes_from_anchors_row: List[int] = [1, 2],
        lanes_from_anchors_col: List[int] = [0, 3],
        valid_thresh: int = 4, 
        pooled_feat_ch: int = 8,
        mlp_mid_dim: int = 2048,
        fc_norm: bool = False,
        feat_idx: int = -1,
        name: str = "lane_kpts_head",
        loc_loss=dict(type="SoftmaxFocalLoss", gamma=2, ignore_lb=-1, reduction='sum', loss_weight=1.0),
        ext_loss=dict(type="CrossEntropyLoss", use_sigmoid=False, reduction="mean", loss_weight=1.0),
    ):
        
        super(UltraFastLaneDetectionV2, self).__init__(name=name)
        self.num_anchors_row = num_anchors_row
        self.num_bins_row = num_bins_row
        self.num_anchors_col = num_anchors_col
        self.num_bins_col = num_bins_col
        self.num_lanes_on_row = num_lanes_on_row
        self.num_lanes_on_col = num_lanes_on_col
        self.lanes_from_anchors_row = lanes_from_anchors_row
        self.lanes_from_anchors_col = lanes_from_anchors_col
        self.valid_thresh = valid_thresh
        self.feat_idx = feat_idx
        self.in_feature_size = in_feature_size

        self.loc_row_dim = self.num_bins_row * self.num_anchors_row * self.num_lanes_on_row
        self.loc_col_dim = self.num_bins_col * self.num_anchors_col * self.num_lanes_on_col
        self.ext_row_dim = 2 * self.num_anchors_row * self.num_lanes_on_row
        self.ext_col_dim = 2 * self.num_anchors_col * self.num_lanes_on_col
        self.total_dim = self.loc_row_dim + self.loc_col_dim + self.ext_row_dim + self.ext_col_dim

        feat_ch, feat_h, feat_w = in_feature_size
        self.input_dim = feat_h * feat_w * pooled_feat_ch
        
        self.ufldv2_cls_block = torch.nn.Sequential(
            torch.nn.Conv2d(feat_ch, pooled_feat_ch, 1),
            torch.nn.Flatten(),
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )

        self.loc_loss = MODELS.build(loc_loss)
        self.ext_loss = MODELS.build(ext_loss)


    def generate_targets(self, batch_data_samples):

        # normalizing [0-1] lane points for target generation
        scale_y = 1 / batch_data_samples[0].metainfo['batch_input_shape'][0]
        scale_x = 1 / batch_data_samples[0].metainfo['batch_input_shape'][1]
        batch_lanes_gt_keypoints = [
            data_sample.get('gt_lanes_kpts') * torch.tensor([scale_x, scale_y, 1], device=data_sample.get('gt_lanes_kpts').device)
            for data_sample in batch_data_samples
        ]

        batch_size = len(batch_lanes_gt_keypoints)
        num_lanes = batch_lanes_gt_keypoints[0].shape[0]
        device = batch_lanes_gt_keypoints[0].device

        # Initialize output tensors with -1
        labels_row = torch.full((batch_size, self.num_anchors_row, num_lanes), -1, 
                            dtype=torch.long, device=device)
        labels_col = torch.full((batch_size, self.num_anchors_col, num_lanes), -1,
                            dtype=torch.long, device=device)

        # Generate anchor positions [H] and [W]. Both are in normalized [0-1] range
        anchor_rows = torch.linspace(0, 1, steps=self.num_anchors_row, device=device)
        anchor_cols = torch.linspace(0, 1, steps=self.num_anchors_col, device=device)

        # Process all batches simultaneously
        for b in range(batch_size):
            if batch_data_samples[b].flip and batch_data_samples[b].flip_direction == 'horizontal':
                # Flip augmentation handles reversing of keypoints (w-x) and (h-y) for horizontal and vertical
                # But we also need to change the lane positions for horizontal flip
                # the left-most lane is now the right-most lane and vice-versa
                # Changing the left <-> right and left_ego <-> right_ego
                batch_lanes_gt_keypoints[b] = batch_lanes_gt_keypoints[b].flip([0])

            # [num_lanes, num_points, 3] -> [num_lanes, num_points] for each component
            x, y, v = batch_lanes_gt_keypoints[b].permute(2, 0, 1)
            
            # Create visibility mask [num_lanes, num_points]
            mask = v > 0
            
            # Count valid points per lane [num_lanes]
            valid_counts = mask.sum(dim=1)
            valid_lanes = valid_counts >= 2
            
            if not torch.any(valid_lanes):
                continue  # Skip if no valid lanes in this batch

            # Process row-wise targets (x at fixed y positions)
            # Sort each lane by y-coordinate for interpolation
            y_sorted, sort_idx = torch.sort(y, dim=1)
            x_sorted = torch.gather(x, 1, sort_idx)
            mask_sorted = torch.gather(mask, 1, sort_idx)
            
            # Get min/max y values for each lane [num_lanes]
            y_min = torch.where(mask_sorted, y_sorted, float('inf')).min(dim=1).values
            y_max = torch.where(mask_sorted, y_sorted, float('-inf')).max(dim=1).values
            
            # Interpolate x at anchor row positions for all lanes
            x_interp = triton_interp(
                anchor_rows.unsqueeze(0).expand(num_lanes, -1),  # [num_lanes, num_anchors_row]
                y_sorted,  # [num_lanes, num_points]
                x_sorted,   # [num_lanes, num_points]
                mask_sorted, # [num_lanes, num_points]
                query_min=y_min.unsqueeze(1),  # [num_lanes, 1]
                query_max=y_max.unsqueeze(1)   # [num_lanes, 1]
            )
            
            # Convert to bin indices
            x_bins = (x_interp * self.num_bins_row).long()
            invalid_mask = (x_bins < 0) | (x_bins >= self.num_bins_row) | torch.isnan(x_interp)
            x_bins[invalid_mask] = -1
            labels_row[b] = x_bins.transpose(0, 1)  # [num_anchors_row, num_lanes]
            
            # Process column-wise targets (y at fixed x positions)
            # Sort each lane by x-coordinate for interpolation
            x_sorted, sort_idx = torch.sort(x, dim=1)
            y_sorted = torch.gather(y, 1, sort_idx)
            mask_sorted = torch.gather(mask, 1, sort_idx)
            
            # Get min/max x values for each lane [num_lanes]
            x_min = torch.where(mask_sorted, x_sorted, float('inf')).min(dim=1).values
            x_max = torch.where(mask_sorted, x_sorted, float('-inf')).max(dim=1).values
            
            # Interpolate y at anchor column positions for all lanes
            y_interp = triton_interp(
                anchor_cols.unsqueeze(0).expand(num_lanes, -1),  # [num_lanes, num_anchors_col]
                x_sorted,  # [num_lanes, num_points]
                y_sorted,  # [num_lanes, num_points]
                mask_sorted, # [num_lanes, num_points]
                query_min=x_min.unsqueeze(1),  # [num_lanes, 1]
                query_max=x_max.unsqueeze(1)   # [num_lanes, 1]
            )
            
            # Convert to bin indices
            y_bins = (y_interp * self.num_bins_col).long()
            invalid_mask = (y_bins < 0) | (y_bins >= self.num_bins_col) | torch.isnan(y_interp)
            y_bins[invalid_mask] = -1
            labels_col[b] = y_bins.transpose(0, 1)  # [num_anchors_col, num_lanes]

        return labels_row, labels_col

    def ufld_v2_losses(
        self, 
        loc_row: torch.Tensor, 
        loc_col: torch.Tensor, 
        ext_row: torch.Tensor, 
        ext_col: torch.Tensor, 
        labels_row: torch.Tensor,
        labels_col: torch.Tensor
    ):

        loc_loss = 0 # localization loss
        ext_loss = 0 # existence loss
        if self.num_bins_row  > 0:
            loc_loss += self.loc_loss(loc_row, labels_row)
            ext_loss += self.ext_loss(ext_row, (labels_row != -1).long())
        if self.num_bins_col  > 0:
            loc_loss += self.loc_loss(loc_col, labels_col)
            ext_loss += self.ext_loss(ext_col, (labels_col != -1).long())

        return {
            "loss/" + self.name + "/loc_loss": loc_loss,
            "loss/" + self.name + "/ext_loss": ext_loss
        }

        
    def forward(self, x):
        y = self.ufldv2_cls_block(x[self.feat_idx])
        B = y.size(0)

        loc_row = y[:, :self.loc_row_dim].view(B, self.num_bins_row, self.num_anchors_row, self.num_lanes_on_row)
        loc_col = y[:, self.loc_row_dim : self.loc_row_dim + self.loc_col_dim].view(B, self.num_bins_col, self.num_anchors_col, self.num_lanes_on_col)
        ext_row = y[:, self.loc_row_dim + self.loc_col_dim : self.loc_row_dim + self.loc_col_dim + self.ext_row_dim].view(B, 2, self.num_anchors_row, self.num_lanes_on_row)
        ext_col = y[:, self.loc_row_dim + self.loc_col_dim + self.ext_row_dim :].view(B, 2, self.num_anchors_col, self.num_lanes_on_col)

        return loc_row, loc_col, ext_row, ext_col

    def loss(self, x: Tuple[torch.Tensor], batch_data_samples: BaseDataElement, **kwargs) -> dict:
        loc_row, loc_col, ext_row, ext_col = self.forward(x)

        labels_row, labels_col = self.generate_targets(batch_data_samples)
        losses_lane_kpts = self.ufld_v2_losses(loc_row, loc_col, ext_row, ext_col, labels_row, labels_col)
        
        return losses_lane_kpts
    
    def predict(
        self,
        x: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[BaseDataElement]]] = None,
        rescale: bool = False
    ) -> List[BaseDataElement]:
        device = x[self.feat_idx].device
        loc_row, loc_col, ext_row, ext_col = self.forward(x)
        
        lanes = torch.zeros((loc_row.shape[0], max(self.num_anchors_row, self.num_anchors_col), max(self.lanes_from_anchors_row + self.lanes_from_anchors_col) + 1, 3), device=loc_row.device)
        
        if self.num_bins_row  > 0:
            ext_row = ext_row.argmax(1)
            bins_row = torch.arange(self.num_bins_row, device=device, dtype=torch.float32).view(1, self.num_bins_row, 1, 1)
            loc_row_xs = (loc_row.softmax(1) * bins_row).sum(1) / self.num_bins_row # normalized row positions
            loc_row_ys = torch.linspace(0, 1, steps=self.num_anchors_row, device=device).view(1, self.num_anchors_row, 1).repeat(x[0].shape[0], 1, self.num_lanes_on_row)
            loc_row = torch.cat([loc_row_xs.unsqueeze(-1), loc_row_ys.unsqueeze(-1), ext_row.unsqueeze(-1)], dim=-1)
            lanes[:, :self.num_anchors_row, self.lanes_from_anchors_row] = loc_row[:, :, self.lanes_from_anchors_row]
        
        if self.num_bins_col:
            ext_col = ext_col.argmax(1)
            bins_col = torch.arange(self.num_bins_col, device=device).view(1, self.num_bins_col, 1, 1)
            loc_col_ys = (loc_col.softmax(1) * bins_col).sum(1) / self.num_bins_col # normalized col positions
            loc_col_xs = torch.linspace(0, 1, steps=self.num_anchors_col, device=device).view(1, self.num_anchors_col, 1).repeat(x[0].shape[0], 1, self.num_lanes_on_col)
            loc_col = torch.stack([loc_col_xs, loc_col_ys, ext_col], dim=-1) 

            lanes[:, :self.num_anchors_col, self.lanes_from_anchors_col] = loc_col[:, :, self.lanes_from_anchors_col]

        predictions = self._get_predictions(lanes, data_samples, rescale)
        return predictions

    def _get_predictions(
        self, 
        lane_lines: torch.Tensor,
        data_samples: List[BaseDataElement],
        rescale: bool
    ):
        """Post-process the output of head.
        """
        if data_samples is None:
            data_samples = [BaseDataElement() for _ in range(len(lane_lines))]

        for data_sample, lane_line in zip(data_samples, lane_lines):
            if rescale:
                h, w = data_sample.metainfo['batch_input_shape']
                sh, sw = data_sample.metainfo['scale_factor']
                lane_line[:, :, 0] *= w/sw # rescaling x
                lane_line[:, :, 1] *= h/sh # rescaling y
            setattr(data_sample, f'pred_{self.name}', lane_line)

        return data_samples