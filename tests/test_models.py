import pytest
import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lightweight_cnn import LightweightMedicalCNN
from models.attention_network import AttentionMedicalCNN, AttentionBlock
from models.traditional_ml import TraditionalMLModels

class TestLightweightCNN:
    def setup_method(self):
        self.model = LightweightMedicalCNN(num_classes=2)
        self.test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model is not None
        assert hasattr(self.model, 'call')
    
    def test_model_forward_pass(self):
        """Test forward pass through the model"""
        output = self.model(self.test_input, training=False)
        
        assert output.shape == (1, 2)  # batch_size, num_classes
        assert np.all(output >= 0) and np.all(output <= 1)  # probabilities
    
    def test_model_trainable_variables(self):
        """Test that model has trainable variables"""
        trainable_vars = self.model.trainable_variables
        assert len(trainable_vars) > 0

class TestAttentionNetwork:
    def setup_method(self):
        self.attention_block = AttentionBlock(32)
        self.attention_model = AttentionMedicalCNN(num_classes=2)
        self.test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    
    def test_attention_block(self):
        """Test attention block functionality"""
        output = self.attention_block(self.test_input)
        
        assert output.shape == self.test_input.shape
    
    def test_attention_model_forward(self):
        """Test forward pass through attention model"""
        output = self.attention_model(self.test_input, training=False)
        
        assert output.shape == (1, 2)
        assert np.all(output >= 0) and np.all(output <= 1)

class TestTraditionalML:
    def setup_method(self):
        self.ml_models = TraditionalMLModels()
        self.X_train = np.random.random((100, 50))  # 100 samples, 50 features
        self.y_train = np.random.randint(0, 2, 100)  # binary classification
    
    def test_model_training(self):
        """Test training traditional ML models"""
        self.ml_models.train(self.X_train, self.y_train, model_name='random_forest')
        
        assert 'random_forest' in self.ml_models.trained_models
    
    def test_model_prediction(self):
        """Test prediction with trained models"""
        self.ml_models.train(self.X_train, self.y_train, model_name='random_forest')
        
        X_test = np.random.random((10, 50))
        predictions = self.ml_models.predict(X_test, model_name='random_forest')
        
        assert predictions.shape == (10, 2)  # 10 samples, 2 classes
        assert np.all(predictions >= 0) and np.all(predictions <= 1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
