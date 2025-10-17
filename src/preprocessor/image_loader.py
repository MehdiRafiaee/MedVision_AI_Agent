# src/preprocessor/image_loader.py
import cv2
import numpy as np
from pathlib import Path

class MedicalImageLoader:
    def __init__(self):
        self.supported_formats = ['.dcm', '.png', '.jpg', '.jpeg', '.tiff']
    
    def load_dicom(self, file_path):
        """بارگذاری فایل‌های DICOM"""
        try:
            import pydicom
            dicom = pydicom.dcmread(file_path)
            image = dicom.pixel_array
            return self.normalize_dicom(image)
        except ImportError:
            raise ImportError("pydicom required for DICOM files")
    
    def load_image(self, file_path):
        """بارگذاری تصویر با پشتیبانی از فرمت‌های مختلف"""
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.dcm':
            return self.load_dicom(file_path)
        else:
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Cannot load image: {file_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def normalize_dicom(self, image):
        """نرمال‌سازی تصاویر DICOM"""
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return (image * 255).astype(np.uint8)
