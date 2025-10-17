import pytest
import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessor.image_loader import MedicalImageLoader
from preprocessor.enhancement import MedicalImageEnhancer

class TestMedicalImageLoader:
    def setup_method(self):
        self.loader = MedicalImageLoader()
        self.test_image_path = "tests/test_data/sample_image.jpg"
        
        # Create test image if doesn't exist
        os.makedirs("tests/test_data", exist_ok=True)
        if not os.path.exists(self.test_image_path):
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(self.test_image_path, test_image)
    
    def test_load_image_success(self):
        """Test successful image loading"""
        image = self.loader.load_image(self.test_image_path)
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
    
    def test_load_image_nonexistent(self):
        """Test loading non-existent image"""
        with pytest.raises(ValueError):
            self.loader.load_image("nonexistent.jpg")
    
    def test_normalize_dicom(self):
        """Test DICOM normalization"""
        test_array = np.random.randint(0, 1000, (100, 100), dtype=np.uint16)
        normalized = self.loader.normalize_dicom(test_array)
        
        assert normalized.dtype == np.uint8
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 255

class TestMedicalImageEnhancer:
    def setup_method(self):
        self.enhancer = MedicalImageEnhancer()
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_enhance_contrast(self):
        """Test contrast enhancement"""
        enhanced = self.enhancer.enhance_contrast(self.test_image)
        assert enhanced.shape == self.test_image.shape
        assert enhanced.dtype == self.test_image.dtype
    
    def test_remove_noise(self):
        """Test noise removal"""
        denoised = self.enhancer.remove_noise(self.test_image)
        assert denoised.shape == self.test_image.shape
    
    def test_adjust_gamma(self):
        """Test gamma adjustment"""
        gamma_corrected = self.enhancer.adjust_gamma(self.test_image, gamma=1.5)
        assert gamma_corrected.shape == self.test_image.shape

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
