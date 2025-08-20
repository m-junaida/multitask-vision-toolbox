# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Any, Optional, Union, Dict, Tuple, List

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import Tensor

from mmdet.utils import ConfigType, OptConfigType
from mmengine.registry import MODELS
from data_structures import MultiTaskDataSample
from mmdet.models.detectors.base import BaseDetector
from utils.misc import replace_substring_dict_keys

@MODELS.register_module()
class MultiTaskModel(BaseDetector):
    r"""Generic class for Multi Task Detector
    we can add any number of heads for different tasks

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        heads: Dict[str, Dict[str, Dict[str, Any]]]. the other heads for multi-task
            Sample:
                {
                    'head_name': {
                        arch: ConfigType,
                        in_variables: List[str],   This should be one of ['back_x', 'neck_x', 'teacher_back_x', 'teacher_neck_x']
                        gt_variables: List[str],   This can be any key we generate in Dataloading stage. TODO: Currently not used. But will be used in future
                        l_kwargs: Dict[str, Any],  kwargs for loss, if any head takes some additional arguments for loss
                        p_kwargs: Dict[str, Any],  kwargs for predict, if any head takes some additional arguments for predict
                        predict_mode: bool         If True this head will also be used in prediction. This is false for kd_head because it is only used for training
                        task_type: object_detection | lane_detection | classification | keypoint | segmentation
                    } 
                }
        teacher_config (:obj:`ConfigDict` | dict | str | Path): Config file
            path or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
            Defaults to True.
        eval_teacher (bool): Set the train mode for teacher.
            Defaults to True.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
    """

    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType = None,
        heads: Dict[str, Dict[str, Dict[str, Any]]] = {},
        teacher_config: Union[ConfigType, str, Path] = None,
        teacher_ckpt: Optional[str] = None,
        eval_teacher: bool = True,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
    ) -> None:
        super(MultiTaskModel, self).__init__(
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        
        self.eval_teacher = eval_teacher        
        self.teacher_model = None
        if teacher_config is not None:
            # Build teacher model
            if isinstance(teacher_config, (str, Path)):
                teacher_config = Config.fromfile(teacher_config)
            self.teacher_model = MODELS.build(teacher_config['model'])
            if teacher_ckpt is not None:
                load_checkpoint(
                    self.teacher_model, teacher_ckpt, map_location='cpu')

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.heads = heads
        for name, head in self.heads.items():
            setattr(self, name, MODELS.build(head['arch']))

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    # This function is just here for overwriting during ONNX export
    def forward(self,
        inputs: torch.Tensor,
        data_samples: Optional[List[MultiTaskDataSample]] = None,
        mode: str = 'tensor'):
        return super().forward(inputs, data_samples, mode)

    def extract_feat(self, batch_inputs: Tensor, return_both=False) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            list(tuple[Tensor]): Multi-level features that may have
            different resolutions for backbone and neck as well
        """
        x = self.backbone(batch_inputs)
        neck_x = None
        if self.with_neck:
            neck_x = self.neck(x)
        
        if return_both:
            return x, neck_x
        
        return neck_x if neck_x is not None else x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: List[MultiTaskDataSample]) -> dict:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        back_x, neck_x = self.extract_feat(batch_inputs, return_both=True)
        
        if self.teacher_model:
            with torch.no_grad():
                teacher_back_x, teacher_neck_x = self.teacher_model.extract_feat(batch_inputs, return_both=True)

        losses = dict()
        for name, head in self.heads.items():
            in_args = []
            for var_name in head['in_variables']:
                in_args.append(locals()[var_name])
            loss = getattr(self, name).loss(*in_args, batch_data_samples, **head.get('l_kwargs', {}))
            loss = replace_substring_dict_keys(loss, 'loss_', 'loss/') # Just something for tensorboard. Losses will be grouped under loss tab
            losses.update(loss)

        return losses

    def predict(self, 
                batch_inputs: Tensor,
                data_samples: Optional[List[MultiTaskDataSample]] = None,
                **kwargs) -> List[MultiTaskDataSample]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        back_x, neck_x = self.extract_feat(batch_inputs, return_both=True)

        if data_samples is None:
            data_samples = [MultiTaskDataSample() for _ in range(batch_inputs.size(0))]

        for name, head in self.heads.items():
            if head.get('predict_mode', False):
                in_args = []
                for var_name in head['in_variables']:
                    in_args.append(locals()[var_name])

                head_out = getattr(self, name).predict(*in_args, data_samples, **head.get('p_kwargs', {}))
                #if head.get('p_kwargs', {}).get('rescale', False): # Rescaling is true then cropping should also be handled
                #    adjust_cropping(head['task_type'], head_out, data_samples)
                if head['task_type'] == 'object_detection':
                    data_samples = self.add_pred_to_datasample(data_samples, head_out)

        return data_samples

    def _forward(self, 
                batch_inputs: Tensor,
                data_samples: Optional[List[MultiTaskDataSample]] = None,
                **kwargs) -> Dict[str, List[Tensor]]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        back_x, neck_x = self.extract_feat(batch_inputs, return_both=True)
        results = dict()
        for name, head in self.heads.items():
            in_args = []
            for var_name in head['in_variables']:
                in_args.append(locals()[var_name])
            head_out = getattr(self, name).forward(*in_args)
            results[name] = head_out

        return results

    def cuda(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling ``cuda`` function."""
        if self.teacher_model:
            self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to other device when calling ``to``
        function."""
        if self.teacher_model:
            self.teacher_model.to(device=device)
        return super().to(device=device)

    def train(self, mode: bool = True) -> None:
        """Set the same train mode for teacher and student model."""
        if self.teacher_model:
            if self.eval_teacher:
                self.teacher_model.train(False)
            else:
                self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

