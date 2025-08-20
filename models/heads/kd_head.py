from typing import Tuple
import torch
from mmengine.registry import MODELS
from mmengine.structures import BaseDataElement
from torch import nn
from models.heads.base_head import BaseHead

@MODELS.register_module()
class KDHead(BaseHead):
    """
    Knowledge distillation head

    Args:
        student_feats_channels (list): List of student feature channels
        teacher_feats_channels (list): List of teacher feature channels
        name (str): Name of the head
        loss_kd (dict): Config for knowledge distillation loss

    Example:
        >>> head = KDHead(
        >>>     student_feats_channels=[128, 256, 512],
        >>>     teacher_feats_channels=[256, 512, 1024],
        >>>     name="kd_head",
        >>>     loss_kd=dict(type="MSELoss", reduction="sum", loss_weight=0.00025),
        >>> )

    """
    def __init__(
        self,
        student_feats_channels=[128, 256, 512],
        teacher_feats_channels=[256, 512, 1024],
        name="kd_head",
        loss_kd=dict(type="MSELoss", reduction="sum", loss_weight=0.00025),
    ):

        assert len(student_feats_channels) == len(teacher_feats_channels), "length of student_feats_channels and teacher_feats_channels should be the same"
        super().__init__(name=name)
        self.loss_kd = MODELS.build(loss_kd)
        
        self.feats_adapter = nn.ModuleList(
            [
                nn.Conv2d(
                    student_feats_channels[i],
                    teacher_feats_channels[i],
                    1,
                    1,
                    "same",
                )
                if student_feats_channels[i] != teacher_feats_channels[i] else nn.Identity()
                for i in range(len(student_feats_channels))
            ]
        )

    def loss(self, x: Tuple[torch.Tensor], teach_x: Tuple[torch.Tensor], batch_data_samples: BaseDataElement, **kwargs):
        loss = torch.zeros(len(x))
        for i in range(len(x)):
            loss[i] = self.loss_kd(self.feats_adapter[i](x[i]), teach_x[i]) / x[i].shape[0]

        return { "loss/" + self.name: torch.sum(loss) }
    