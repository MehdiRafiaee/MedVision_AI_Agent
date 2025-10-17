import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: np.ndarray = None) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            # ROC AUC if probabilities are provided
            if y_prob is not None:
                try:
                    if y_prob.shape[1] == 2:  # Binary classification
                        metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    else:  # Multi-class
                        metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                except Exception as e:
                    logger.warning(f"Could not calculate AUC: {e}")
                    metrics['auc'] = 0.0
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics['classification_report'] = report
            
            # Additional metrics
            metrics = self._calculate_additional_metrics(metrics, y_true, y_pred)
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _calculate_additional_metrics(self, metrics: Dict, y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict:
        """Calculate additional specialized metrics"""
        # Sensitivity and Specificity for binary classification
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['precision_pos'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['neg_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return metrics
    
    def calculate_confidence_intervals(self, metrics: Dict, n_bootstraps: int = 1000) -> Dict:
        """Calculate bootstrap confidence intervals for metrics"""
        # This would require the original predictions and labels
        # Implementation depends on specific use case
        pass
    
    def compare_models(self, metrics_list: List[Dict], model_names: List[str]) -> Dict:
        """Compare metrics across multiple models"""
        comparison = {}
        
        for i, metrics in enumerate(metrics_list):
            model_name = model_names[i] if i < len(model_names) else f"Model_{i+1}"
            comparison[model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'auc': metrics.get('auc', 0)
            }
        
        return comparison
