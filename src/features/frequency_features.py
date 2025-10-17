# src/features/frequency_features.py
import pywt
import numpy as np

class FrequencyFeatureExtractor:
    def __init__(self, wavelet='db4', level=3):
        self.wavelet = wavelet
        self.level = level
    
    def calculate_wavelet_features(self, image):
        """استخراج ویژگی‌های حوزه فرکانس با wavelet transform"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # wavelet decomposition
        coeffs = pywt.wavedec2(image, self.wavelet, level=self.level)
        
        features = []
        # ویژگی‌های انرژی در هر سطح
        for i, coeff in enumerate(coeffs):
            if i == 0:
                # approximate coefficients
                energy = np.sum(coeff**2)
                features.extend([energy, np.mean(coeff), np.std(coeff)])
            else:
                # detail coefficients
                for detail in coeff:
                    energy = np.sum(detail**2)
                    features.extend([energy, np.mean(detail), np.std(detail)])
        
        return features
    
    def calculate_fourier_features(self, image):
        """ویژگی‌های Fourier Transform"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # تقسیم به مناطق فرکانسی
        h, w = magnitude_spectrum.shape
        regions = [
            magnitude_spectrum[:h//2, :w//2],  # پایین-چپ (فرکانس پایین)
            magnitude_spectrum[:h//2, w//2:],  # پایین-راست
            magnitude_spectrum[h//2:, :w//2],  # بالا-چپ
            magnitude_spectrum[h//2:, w//2:]   # بالا-راست (فرکانس بالا)
        ]
        
        features = []
        for region in regions:
            features.extend([np.mean(region), np.std(region), np.max(region)])
        
        return features
