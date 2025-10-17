import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.texture_features import TextureFeatureExtractor
from features.shape_features import ShapeFeatureExtractor
from features.frequency_features import FrequencyFeatureExtractor

class TestTextureFeatures:
    def setup_method(self):
        self.extractor = TextureFeatureExtractor()
        self.test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    def test_glcm_features(self):
        """Test GLCM feature extraction"""
        features = self.extractor.calculate_glcm_features(self.test_image)
        
        expected_features = ['contrast', 'dissimilarity', 'homogeneity', 
                           'energy', 'correlation', 'ASM']
        
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], float)
    
    def test_lbp_features(self):
        """Test LBP feature extraction"""
        features = self.extractor.calculate_lbp_features(self.test_image)
        
        assert isinstance(features, list)
        assert len(features) == 10  # 8 points + 2
        assert all(isinstance(x, float) for x in features)

class TestShapeFeatures:
    def setup_method(self):
        self.extractor = ShapeFeatureExtractor()
        # Create a test image with a simple shape
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(self.test_image, (20, 20), (80, 80), 255, -1)
    
    def test_hu_moments(self):
        """Test Hu moments calculation"""
        moments = self.extractor.calculate_hu_moments(self.test_image)
        
        assert isinstance(moments, list)
        assert len(moments) == 7
        assert all(isinstance(m, float) for m in moments)
    
    def test_contour_features(self):
        """Test contour features calculation"""
        features = self.extractor.calculate_contour_features(self.test_image)
        
        assert isinstance(features, list)
        assert len(features) == 5
        assert all(isinstance(f, float) for f in features)

class TestFrequencyFeatures:
    def setup_method(self):
        self.extractor = FrequencyFeatureExtractor()
        self.test_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    
    def test_wavelet_features(self):
        """Test wavelet feature extraction"""
        features = self.extractor.calculate_wavelet_features(self.test_image)
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, float) for f in features)
    
    def test_fourier_features(self):
        """Test Fourier feature extraction"""
        features = self.extractor.calculate_fourier_features(self.test_image)
        
        assert isinstance(features, list)
        assert len(features) == 12  # 4 regions * 3 features
        assert all(isinstance(f, float) for f in features)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
