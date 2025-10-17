# src/features/texture_features.py
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

class TextureFeatureExtractor:
    def __init__(self):
        self.distances = [1, 2, 3]
        self.angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    def calculate_glcm_features(self, image):
        """محاسبه ویژگی‌های GLCM"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # نرمال‌سازی به ۸ سطح
        image = (image / 32).astype(np.uint8)
        
        glcm = greycomatrix(image, distances=self.distances, angles=self.angles, 
                           levels=8, symmetric=True, normed=True)
        
        features = {}
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in properties:
            features[prop] = greycoprops(glcm, prop).mean()
        
        return features
    
    def calculate_lbp_features(self, image, points=8, radius=1):
        """محاسبه Local Binary Pattern"""
        from skimage.feature import local_binary_pattern
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        lbp = local_binary_pattern(image, points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=points+2, range=(0, points+2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # نرمال‌سازی
        
        return hist.tolist()
