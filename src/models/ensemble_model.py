import tensorflow as tf
import numpy as np
from typing import Dict, List, Union
import logging

from .lightweight_cnn import LightweightMedicalCNN
from .attention_network import AttentionMedicalCNN
from .traditional_ml import TraditionalMLModels

logger = logging.getLogger(__name__)

class EnsembleMedicalModel:
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.model_weights = {}
        self.is_trained = False
        
    def initialize_models(self):
        """Initialize all model types"""
        try:
            # Deep Learning Models
            self.models['lightweight_cnn'] = LightweightMedicalCNN(
                num_classes=self.config['model']['num_classes']
            )
            
            self.models['attention_cnn'] = AttentionMedicalCNN(
                num_classes=self.config['model']['num_classes']
            )
            
            # Traditional ML Models
            self.models['traditional_ml'] = TraditionalMLModels()
            
            # Initialize weights (can be learned or set manually)
            self.model_weights = {
                'lightweight_cnn': 0.4,
                'attention_cnn': 0.4, 
                'traditional_ml': 0.2
            }
            
            logger.info("✅ All models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
            raise
    
    def train_ensemble(self, train_data, val_data, feature_data=None, labels=None):
        """Train all models in the ensemble"""
        try:
            # Train CNN models
            if 'lightweight_cnn' in self.models:
                logger.info("🏋️ Training Lightweight CNN...")
                self._train_cnn_model('lightweight_cnn', train_data, val_data)
            
            if 'attention_cnn' in self.models:
                logger.info("🏋️ Training Attention CNN...")  
                self._train_cnn_model('attention_cnn', train_data, val_data)
            
            # Train traditional models if feature data is provided
            if 'traditional_ml' in self.models and feature_data is not None and labels is not None:
                logger.info("🏋️ Training Traditional ML models...")
                self.models['traditional_ml'].train(feature_data, labels)
            
            self.is_trained = True
            logger.info("✅ Ensemble training completed")
            
        except Exception as e:
            logger.error(f"❌ Error in ensemble training: {e}")
            raise
    
    def _train_cnn_model(self, model_name: str, train_data, val_data):
        """Train individual CNN model"""
        model = self.models[model_name]
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['model']['learning_rate']
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=self.config['training']['reduce_lr_patience']
            )
        ]
        
        # Train model
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config['model']['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X, feature_data=None) -> Dict:
        """Make prediction using ensemble"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        try:
            predictions = {}
            
            # CNN model predictions
            if 'lightweight_cnn' in self.models:
                cnn_pred = self.models['lightweight_cnn'].predict(X)
                predictions['lightweight_cnn'] = cnn_pred
            
            if 'attention_cnn' in self.models:
                attention_pred = self.models['attention_cnn'].predict(X)  
                predictions['attention_cnn'] = attention_pred
            
            # Traditional model predictions
            if 'traditional_ml' in self.models and feature_data is not None:
                traditional_pred = self.models['traditional_ml'].predict(
                    feature_data, 'random_forest'
                )
                predictions['traditional_ml'] = traditional_pred
            
            # Ensemble prediction (weighted average)
            ensemble_pred = self._combine_predictions(predictions)
            
            return {
                'ensemble_prediction': ensemble_pred,
                'individual_predictions': predictions,
                'confidence': np.max(ensemble_pred)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in ensemble prediction: {e}")
            raise
    
    def _combine_predictions(self, predictions: Dict) -> np.ndarray:
        """Combine predictions using weighted averaging"""
        total_weight = 0
        weighted_sum = None
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 0.1)
            
            if weighted_sum is None:
                weighted_sum = weight * pred
            else:
                weighted_sum += weight * pred
            
            total_weight += weight
        
        # Normalize by total weight
        ensemble_pred = weighted_sum / total_weight
        
        return ensemble_pred
    
    def evaluate_ensemble(self, test_data, feature_data=None, labels=None) -> Dict:
        """Comprehensive evaluation of ensemble performance"""
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        # Get predictions
        results = self.predict(test_data, feature_data)
        ensemble_pred = results['ensemble_prediction']
        
        # Convert to class predictions
        y_pred = np.argmax(ensemble_pred, axis=1)
        y_true = np.concatenate([y for x, y in test_data], axis=0)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC AUC
        try:
            auc = roc_auc_score(y_true, ensemble_pred[:, 1])
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'auc_score': auc,
            'individual_results': results['individual_predictions']
        }
    
    def get_model_contributions(self) -> Dict:
        """Get contribution of each model to final prediction"""
        return self.model_weights
    
    def save_ensemble(self, filepath: str):
        """Save ensemble models"""
        import joblib
        
        ensemble_data = {
            'model_weights': self.model_weights,
            'config': self.config
        }
        
        # Save ensemble metadata
        joblib.dump(ensemble_data, f"{filepath}_ensemble_metadata.joblib")
        
        # Save individual models
        for model_name, model in self.models.items():
            if hasattr(model, 'save'):
                model.save(f"{filepath}_{model_name}.h5")
            else:
                joblib.dump(model, f"{filepath}_{model_name}.joblib")
        
        logger.info(f"✅ Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble models"""
        import joblib
        
        # Load ensemble metadata
        ensemble_data = joblib.load(f"{filepath}_ensemble_metadata.joblib")
        self.model_weights = ensemble_data['model_weights']
        self.config = ensemble_data['config']
        
        # Load individual models
        self.initialize_models()
        
        for model_name in self.models.keys():
            try:
                if model_name.endswith('cnn'):
                    self.models[model_name] = tf.keras.models.load_model(
                        f"{filepath}_{model_name}.h5"
                    )
                else:
                    self.models[model_name] = joblib.load(
                        f"{filepath}_{model_name}.joblib"
                    )
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")
        
        self.is_trained = True
        logger.info(f"✅ Ensemble loaded from {filepath}")
