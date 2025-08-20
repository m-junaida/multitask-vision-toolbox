from typing import List, Optional, Union
import torch
from mmengine.structures import LabelData

class MultiLabelData(LabelData):
    """Data structure for multi-label classification.

    Attributes:
        labels (Tensor): 1D tensor of label indices (e.g., [0, 2, 4]).
        scores (Tensor): Raw class scores (after sigmoid).
        num_classes (int): Total number of possible classes.
    """

    def __init__(
        self,
        labels: Optional[Union[List[int], torch.Tensor]] = None,
        scores: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        if labels is not None:
            self.labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        if scores is not None:
            self.scores = scores
        if num_classes is not None:
            self.num_classes = num_classes

    @classmethod
    def from_labels(cls, labels: List[int], num_classes: int=None) -> 'MultiLabelData':
        """Create from a list of label indices."""
        return cls(labels=labels, num_classes=num_classes)

    @classmethod
    def from_scores(
        cls,
        scores: torch.Tensor,
        threshold: Union[float, List[float], torch.Tensor] = 0.5
    ) -> 'MultiLabelData':
        """Create from sigmoid scores with threshold(s).

        Args:
            scores (Tensor): Shape (num_classes,).
            threshold (float | List[float] | Tensor): Scalar or per-class thresholds.
        """
        if isinstance(threshold, list):
            threshold = torch.tensor(threshold, dtype=scores.dtype, device=scores.device)
        elif isinstance(threshold, float):
            threshold = torch.full_like(scores, threshold)
        elif not isinstance(threshold, torch.Tensor):
            raise TypeError(f"Unsupported threshold type: {type(threshold)}")

        labels = (scores > threshold).nonzero(as_tuple=False).squeeze(1)
        return cls(labels=labels, scores=scores, num_classes=scores.size(0))

    def to_onehot(self) -> torch.Tensor:
        """Return one-hot/multi-hot vector."""
        assert hasattr(self, 'labels'), "No labels found to convert."
        assert hasattr(self, 'num_classes'), "num_classes must be set for one-hot conversion."
        onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        onehot[self.labels] = 1.0
        return onehot

    def to_labels(
        self,
        threshold: Union[float, List[float], torch.Tensor] = 0.5
    ) -> torch.Tensor:
        """Get label indices from scores using threshold(s).

        Args:
            threshold (float | List[float] | Tensor): Threshold for each class.
        """
        if hasattr(self, 'scores') and self.scores is not None:
            if isinstance(threshold, list):
                threshold = torch.tensor(threshold, dtype=self.scores.dtype, device=self.scores.device)
            elif isinstance(threshold, float):
                threshold = torch.full_like(self.scores, threshold)
            elif not isinstance(threshold, torch.Tensor):
                raise TypeError(f"Unsupported threshold type: {type(threshold)}")

            return (self.scores > threshold).nonzero(as_tuple=False).squeeze(1)
        elif hasattr(self, 'labels'):
            return self.labels
        else:
            raise ValueError("Neither scores nor labels are set.")

    def __repr__(self) -> str:
        return f"MultiLabelData(labels={getattr(self, 'labels', None)}, " \
               f"scores={getattr(self, 'scores', None)}, " \
               f"num_classes={getattr(self, 'num_classes', None)})"

