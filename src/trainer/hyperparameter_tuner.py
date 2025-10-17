import tensorflow as tf
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, config: dict):
        self.config = config
        
    def tune_cnn_hyperparameters(self, model_fn, train_data, val_data, 
                               param_grid: Dict) -> Dict:
        """Tune hyperparameters for CNN models"""
        best_score = 0
        best_params = {}
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        logger.info(f"🔍 Tuning {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Create model with current parameters
                model = model_fn(**params)
                
                # Compile and train
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=20,  # Short training for tuning
                    verbose=0
                )
                
                # Get validation accuracy
                val_accuracy = max(history.history['val_accuracy'])
                
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = params.copy()
                    
                logger.info(f"Validation accuracy: {val_accuracy:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                continue
        
        logger.info(f"🎯 Best parameters: {best_params} (score: {best_score:.4f})")
        return best_params
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all combinations of parameters"""
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def random_search(self, model_fn, train_data, val_data, 
                     param_distributions: Dict, n_iter: int = 10) -> Dict:
        """Random search for hyperparameter optimization"""
        best_score = 0
        best_params = {}
        
        for i in range(n_iter):
            # Sample random parameters
            params = self._sample_parameters(param_distributions)
            logger.info(f"Random search {i+1}/{n_iter}: {params}")
            
            try:
                model = model_fn(**params)
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=15,
                    verbose=0
                )
                
                val_accuracy = max(history.history['val_accuracy'])
                
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = params.copy()
                    
            except Exception as e:
                logger.warning(f"Random search failed: {e}")
                continue
        
        return best_params
    
    def _sample_parameters(self, param_distributions: Dict) -> Dict:
        """Sample parameters from distributions"""
        import random
        
        sampled_params = {}
        
        for param, distribution in param_distributions.items():
            if isinstance(distribution, list):
                sampled_params[param] = random.choice(distribution)
            elif isinstance(distribution, tuple) and len(distribution) == 2:
                # Continuous range
                low, high = distribution
                if isinstance(low, int):
                    sampled_params[param] = random.randint(low, high)
                else:
                    sampled_params[param] = random.uniform(low, high)
        
        return sampled_params
