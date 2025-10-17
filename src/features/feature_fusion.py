import numpy as np
import pandas as pd
from typing import Dict, List, Union
import logging
from pathlib import Path

from .texture_features import TextureFeatureExtractor
from .shape_features import ShapeFeatureExtractor
from .frequency_features import FrequencyFeatureExtractor

logger = logging.getLogger(__name__)

class FeatureFusion:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.texture_extractor = TextureFeatureExtractor()
        self.shape_extractor = ShapeFeatureExtractor()
        self.frequency_extractor = FrequencyFeatureExtractor()
        
        self.feature_names = []
        
    def extract_all_features(self, image: np.ndarray) -> Dict[str, Union[float, List[float]]]:
        """Extract all features from different modalities"""
        try:
            features = {}
            
            # Texture features
            texture_features = self.texture_extractor.calculate_glcm_features(image)
            features.update(texture_features)
            
            lbp_features = self.texture_extractor.calculate_lbp_features(image)
            features['lbp_features'] = lbp_features
            
            # Shape features
            hu_moments = self.shape_extractor.calculate_hu_moments(image)
            features['hu_moments'] = hu_moments
            
            contour_features = self.shape_extractor.calculate_contour_features(image)
            features['contour_features'] = contour_features
            
            # Frequency features
            wavelet_features = self.frequency_extractor.calculate_wavelet_features(image)
            features['wavelet_features'] = wavelet_features
            
            fourier_features = self.frequency_extractor.calculate_fourier_features(image)
            features['fourier_features'] = fourier_features
            
            # Flatten all features
            flattened_features = self._flatten_features(features)
            
            return flattened_features
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return {}
    
    def _flatten_features(self, features: Dict) -> Dict[str, float]:
        """Flatten nested feature dictionaries"""
        flattened = {}
        
        for key, value in features.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened[f"{key}_{sub_key}"] = sub_value
            elif isinstance(value, (list, np.ndarray)):
                for i, item in enumerate(value):
                    flattened[f"{key}_{i}"] = item
            else:
                flattened[key] = value
        
        self.feature_names = list(flattened.keys())
        return flattened
    
    def extract_features_batch(self, image_paths: List[str], labels: List[str] = None) -> pd.DataFrame:
        """Extract features for a batch of images"""
        features_list = []
        valid_labels = []
        
        for i, image_path in enumerate(image_paths):
            try:
                import cv2
                
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                # Extract features
                features = self.extract_all_features(image)
                features['image_path'] = str(image_path)
                
                if labels is not None:
                    features['label'] = labels[i]
                    valid_labels.append(labels[i])
                
                features_list.append(features)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} images...")
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(features_list)
        
        logger.info(f"Successfully extracted features from {len(features_list)} images")
        return df
    
    def select_important_features(self, features_df: pd.DataFrame, target_column: str = 'label', 
                                n_features: int = 50) -> List[str]:
        """Select most important features using Random Forest"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        
        # Separate features and target
        X = features_df.drop(columns=[target_column, 'image_path'], errors='ignore')
        y = features_df[target_column]
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Select features based on importance
        selector = SelectFromModel(rf, prefit=True, max_features=n_features)
        selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} important features")
        return selected_features
    
    def normalize_features(self, features_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Normalize features to [0, 1] range"""
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(features_df[feature_columns])
        
        # Create new DataFrame with normalized features
        normalized_df = features_df.copy()
        normalized_df[feature_columns] = features_normalized
        
        return normalized_df, scaler
