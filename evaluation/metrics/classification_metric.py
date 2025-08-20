from typing import Sequence, Optional
import pandas as pd
import torch
from mmpretrain.evaluation import MultiLabelMetric
from mmpretrain.registry import METRICS

@METRICS.register_module()
class ClassificationMetric(MultiLabelMetric):

    default_prefix: Optional[str] = 'classification'

    def __init__(self, 
                 gt_name="gt_classification_labels", 
                 pred_name="pred_classification_head", 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_name = gt_name
        self.pred_name = pred_name
        
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        
        for data_sample in data_samples:
            assert self.gt_name in data_sample, "{} not in data_sample".format(self.gt_name)
            assert self.pred_name in data_sample, "{} not in data_sample".format(self.pred_name)
            
            # Re-formating how parent class (MultiLabelMetric) expects data
            self.results.append(dict(
                gt_score=data_sample[self.gt_name].clone(), 
                pred_score=data_sample[self.pred_name].clone()
            ))

    def evaluate(self, size):
        target = torch.stack([res['gt_score'] for res in self.results])
        target = target.detach().to(device=self.collect_device).numpy()
        metrics = super().evaluate(size)

        # Print classwise metrics
        if self.average is None and self.dataset_meta.get(f'{self.prefix}_classes', None) is not None:
            metrics_df = pd.DataFrame(metrics, index=self.dataset_meta[f'{self.prefix}_classes'])
            metrics_df.columns = ['precision', 'recall', 'f1-score']
            metrics_df['#samples'] = target.sum(0)

            ## Computing weighted averages
            class_weights = (target.sum(0) / target.shape[0])
            # normalizing weights because the sum of the weights should be 1
            class_weights /= class_weights.sum()
            weighted_average = {}
            for k in ['precision', 'recall', 'f1-score']:
                metric_k = f'{self.prefix}/{k}_classwise'
                weighted_average[k] = sum(cw * cm for cw, cm in zip(class_weights, metrics[metric_k]))
                metrics[f'{self.prefix}/weighted_avg_{k}'] = weighted_average[k]
            
            weighted_average['#samples'] = target.shape[0]
            metrics_df = pd.concat([metrics_df, pd.DataFrame(weighted_average, index=['weighted_average'])])
            
            print(metrics_df.to_string())


        return metrics