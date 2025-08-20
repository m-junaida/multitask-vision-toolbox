from typing import Tuple, Optional, List
import torch
from torch import Tensor, nn

from mmdet.structures import SampleList
from utils.misc import stack_data_samples
from mmengine.registry import MODELS
from mmengine.structures import BaseDataElement
from models.heads.base_head import BaseHead

@MODELS.register_module()
class ClassificationHead(BaseHead):
    def __init__(
        self,
        in_channels=512,
        classes=2,
        name="classification_head",
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=2.0)
    ):
        super().__init__(name=name)
        self.loss_cls = MODELS.build(loss_cls)
        
        self.classification_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, classes),
        )

    def forward(self, x: Tuple[torch.Tensor]) -> Tensor:
        return self.classification_block(x[-1])
    
    def loss(self, x: Tuple[torch.Tensor], batch_data_samples: SampleList, **kwargs) -> dict:
        targets = stack_data_samples(batch_data_samples, "gt_classification_labels")
        y = self.forward(x)
        return { "loss/" + self.name: self.loss_cls(y, targets) / y.shape[0] }

    def predict(
        self,
        x: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[BaseDataElement]]] = None
    ) -> List[BaseDataElement]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        y = self.forward(x)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(y, data_samples)
        return predictions

    def _get_predictions(self, cls_score: torch.Tensor,
                         data_samples: List[BaseDataElement]):
        """Post-process the output of head.
        """
        pred_scores = torch.sigmoid(cls_score)

        if data_samples is None:
            data_samples = [BaseDataElement() for _ in range(cls_score.size(0))]

        for data_sample, score in zip(data_samples, pred_scores):
            setattr(data_sample, f'pred_{self.name}', score)

        return data_samples

    
