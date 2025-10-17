import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class CrossValidator:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        
    def cross_validate_cnn(self, model_fn, X, y, epochs: int = 50) -> Dict:
        """Perform k-fold cross validation for CNN models"""
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_scores = []
        histories = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            logger.info(f"🏁 Training fold {fold + 1}/{self.n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = model_fn()
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                verbose=0
            )
            
            # Evaluate
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(val_accuracy)
            histories.append(history.history)
            
            logger.info(f"Fold {fold + 1} - Val Accuracy: {val_accuracy:.4f}")
        
        # Calculate statistics
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        logger.info(f"📊 Cross-validation results: {mean_score:.4f} ± {std_score:.4f}")
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'histories': histories
        }
    
    def stratified_cross_validate(self, model_fn, X, y, epochs: int = 50) -> Dict:
        """Stratified k-fold cross validation"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"🏁 Training stratified fold {fold + 1}/{self.n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = model_fn()
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model.fit(X_train, y_train, epochs=epochs, verbose=0)
            
            val_accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
            fold_scores.append(val_accuracy)
            
            logger.info(f"Fold {fold + 1} - Val Accuracy: {val_accuracy:.4f}")
        
        return {
            'fold_scores': fold_scores,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores)
        }
