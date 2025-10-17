# src/preprocessor/enhancement.py
import cv2
import numpy as np

class MedicalImageEnhancer:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    def enhance_contrast(self, image):
        """بهبود کنتراست با CLAHE"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = self.clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            return self.clahe.apply(image)
    
    def remove_noise(self, image):
        """حذف نویز با الگوریتم Non-local Means"""
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )
    
    def adjust_gamma(self, image, gamma=1.2):
        """تنظیم گاما برای بهبود روشنایی"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
