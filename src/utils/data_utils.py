import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config: dict):
        self.config = config
        self.data_info = {}
    
    def load_dataset(self, data_dir: str, file_format: str = "auto") -> pd.DataFrame:
        """Load dataset from directory"""
        data_dir = Path(data_dir)
        
        if file_format == "auto":
            file_format = self._detect_file_format(data_dir)
        
        if file_format == "csv":
            return self._load_csv_data(data_dir)
        elif file_format == "images":
            return self._load_image_data(data_dir)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def _detect_file_format(self, data_dir: Path) -> str:
        """Detect file format automatically"""
        # Check for CSV files
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            return "csv"
        
        # Check for image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.dcm']
        image_files = []
        for ext in image_extensions:
            image_files.extend(data_dir.glob(f"**/{ext}"))
        
        if image_files:
            return "images"
        
        raise ValueError("Could not detect file format")
    
    def _load_csv_data(self, data_dir: Path) -> pd.DataFrame:
        """Load data from CSV files"""
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in directory")
        
        # Load and combine all CSV files
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"✅ Loaded {len(combined_df)} samples from {len(csv_files)} CSV files")
        
        return combined_df
    
    def _load_image_data(self, data_dir: Path) -> pd.DataFrame:
        """Create dataframe from image directory structure"""
        image_data = []
        
        # Assuming directory structure: data_dir/class_name/image_files
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_files = list(class_dir.glob("*.*"))
                
                for img_file in image_files:
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.dcm']:
                        image_data.append({
                            'image_path': str(img_file),
                            'label': class_name,
                            'file_size': img_file.stat().st_size
                        })
        
        df = pd.DataFrame(image_data)
        logger.info(f"✅ Loaded {len(df)} images from {len(set(df['label']))} classes")
        
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                  val_size: float = 0.1, random_state: int = 42) -> Tuple:
        """Split data into train/validation/test sets"""
        from sklearn.model_selection import train_test_split
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['label']
        )
        
        # Second split: separate validation set from train
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=train_val_df['label']
        )
        
        logger.info(f"📊 Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_class_distribution(self, df: pd.DataFrame) -> Dict:
        """Get class distribution statistics"""
        class_counts = df['label'].value_counts()
        class_dist = (class_counts / len(df)).to_dict()
        
        logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            percentage = class_dist[class_name] * 100
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return {
            'counts': class_counts.to_dict(),
            'distribution': class_dist,
            'total_samples': len(df),
            'num_classes': len(class_counts)
        }
    
    def balance_dataset(self, df: pd.DataFrame, method: str = "oversample") -> pd.DataFrame:
        """Balance dataset using various methods"""
        from sklearn.utils import resample
        
        if method == "oversample":
            return self._oversample_minority_classes(df)
        elif method == "undersample":
            return self._undersample_majority_classes(df)
        else:
            logger.warning(f"Unknown balancing method: {method}")
            return df
    
    def _oversample_minority_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oversample minority classes to balance dataset"""
        class_counts = df['label'].value_counts()
        max_count = class_counts.max()
        
        balanced_dfs = []
        
        for class_name in class_counts.index:
            class_df = df[df['label'] == class_name]
            
            if len(class_df) < max_count:
                # Oversample minority class
                oversampled = resample(
                    class_df,
                    replace=True,
                    n_samples=max_count,
                    random_state=42
                )
                balanced_dfs.append(oversampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"✅ Dataset balanced: {len(balanced_df)} samples")
        
        return balanced_df
    
    def _undersample_majority_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undersample majority classes to balance dataset"""
        class_counts = df['label'].value_counts()
        min_count = class_counts.min()
        
        balanced_dfs = []
        
        for class_name in class_counts.index:
            class_df = df[df['label'] == class_name]
            
            if len(class_df) > min_count:
                # Undersample majority class
                undersampled = resample(
                    class_df,
                    replace=False,
                    n_samples=min_count,
                    random_state=42
                )
                balanced_dfs.append(undersampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"✅ Dataset balanced: {len(balanced_df)} samples")
        
        return balanced_df
