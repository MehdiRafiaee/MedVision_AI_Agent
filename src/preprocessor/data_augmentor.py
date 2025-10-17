import numpy as np
import cv2
import tensorflow as tf
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class MedicalDataAugmentor:
    def __init__(self, config: dict):
        self.config = config
        self.augmentation_params = config.get('augmentation', {})
        
    def create_augmentation_pipeline(self):
        """Create TensorFlow data augmentation pipeline"""
        return tf.keras.Sequential([
            tf.keras.layers.RandomRotation(self.augmentation_params.get('rotation_range', 0.2)),
            tf.keras.layers.RandomZoom(self.augmentation_params.get('zoom_range', 0.2)),
            tf.keras.layers.RandomContrast(self.augmentation_params.get('contrast_range', 0.2)),
            tf.keras.layers.RandomTranslation(
                height_factor=self.augmentation_params.get('height_shift_range', 0.2),
                width_factor=self.augmentation_params.get('width_shift_range', 0.2)
            ),
        ])
    
    def augment_image(self, image: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to image and optional mask"""
        try:
            augmented_image = image.copy()
            
            # Random rotation
            if np.random.random() < 0.5:
                angle = np.random.uniform(-20, 20)
                augmented_image = self.rotate_image(augmented_image, angle)
                if mask is not None:
                    mask = self.rotate_image(mask, angle)
            
            # Random flip
            if np.random.random() < 0.5:
                flip_code = np.random.choice([-1, 0, 1])
                augmented_image = cv2.flip(augmented_image, flip_code)
                if mask is not None:
                    mask = cv2.flip(mask, flip_code)
            
            # Random brightness
            if np.random.random() < 0.3:
                augmented_image = self.adjust_brightness(augmented_image)
            
            # Random gamma correction
            if np.random.random() < 0.3:
                gamma = np.random.uniform(0.7, 1.3)
                augmented_image = self.adjust_gamma(augmented_image, gamma)
            
            return augmented_image, mask
            
        except Exception as e:
            logger.error(f"Error in image augmentation: {e}")
            return image, mask
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated_image
    
    def adjust_brightness(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Adjust image brightness"""
        if factor is None:
            factor = np.random.uniform(0.7, 1.3)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Adjust image gamma"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def elastic_transform(self, image: np.ndarray, alpha: float = 100, sigma: float = 10) -> np.ndarray:
        """Apply elastic transformation (for more realistic deformations)"""
        random_state = np.random.RandomState(None)
        shape = image.shape[:2]
        
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1
        
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        return cv2.remap(image, indices[1].astype(np.float32), indices[0].astype(np.float32), 
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
