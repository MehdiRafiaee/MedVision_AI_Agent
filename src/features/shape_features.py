# src/features/shape_features.py
import cv2
import numpy as np

class ShapeFeatureExtractor:
    def __init__(self):
        pass
    
    def calculate_hu_moments(self, image):
        """محاسبه momentهای Hu"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # آستانه‌گذاری برای استخراج شکل
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # محاسبه moments
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments)
        
        # log transform برای مقیاس‌پذیری بهتر
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-8)
        
        return hu_moments.flatten().tolist()
    
    def calculate_contour_features(self, image):
        """محاسبه ویژگی‌های کانتور"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(5).tolist()
        
        contour = max(contours, key=cv2.contourArea)
        
        features = []
        # مساحت
        features.append(cv2.contourArea(contour))
        # محیط
        features.append(cv2.arcLength(contour, True))
        # دایره‌ای بودن
        features.append((4 * np.pi * features[0]) / (features[1] ** 2 + 1e-8))
        # مستطیلی بودن
        _, (w, h), _ = cv2.minAreaRect(contour)
        features.append(min(w, h) / max(w, h) if max(w, h) > 0 else 0)
        # جامعیت
        features.append(features[0] / (w * h + 1e-8))
        
        return features
