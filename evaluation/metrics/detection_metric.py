from typing import Sequence, List, Optional
from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class DetectionMetric(CocoMetric):

    default_prefix: Optional[str] = 'detection'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        
        for data_sample in data_samples:
            # changing data format to how CocoMetric class needs
            if 'instances' not in data_sample:
                data_sample['instances'] = []
                for label, box in zip(data_sample['gt_instances']['labels'], data_sample['gt_instances']['bboxes']):
                    data_sample['instances'].append(dict(bbox=box.cpu().numpy(), bbox_label=label))
            
        # Detection Results will be processed from parent class
        super().process(data_batch, data_samples)
