#!/usr/bin/env python3
"""
MedVision AI Agent - Main Entry Point
Complete from-scratch medical image analysis system
"""

import os
import argparse
import yaml
import numpy as np
from pathlib import Path

# Import TensorFlow here to fix the error
import tensorflow as tf

from src.preprocessor.image_loader import MedicalImageLoader
from src.preprocessor.enhancement import MedicalImageEnhancer
from src.features.feature_fusion import FeatureFusion
from src.models.lightweight_cnn import LightweightMedicalCNN
from src.models.traditional_ml import TraditionalMLModels
from src.trainer.model_trainer import MedicalModelTrainer
from src.evaluator.visualization import ResultVisualizer
from src.utils.logger import setup_logging

def setup_project_directories():
    """Create necessary project directories automatically"""
    
    # Define directory structure
    directories = [
        # Data directories
        "data/raw/covid_ct/covid",
        "data/raw/covid_ct/non_covid",
        "data/raw/chest_xray/normal",
        "data/raw/chest_xray/pneumonia",
        "data/processed/images",
        "data/processed/labels",
        "data/augmented/rotation",
        "data/augmented/flip",
        "data/augmented/brightness",
        
        # Output directories
        "output/models",
        "output/predictions/single",
        "output/predictions/batch",
        "output/reports/training",
        "output/reports/evaluation",
        "output/visualizations/training_curves",
        "output/visualizations/attention_maps",
        "output/visualizations/confusion_matrices",
        "output/visualizations/feature_importance",
        
        # Other directories
        "logs",
        "config"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created successfully!")
    print("📁 data/ - For medical images and datasets")
    print("📊 output/ - For models, predictions, and reports")
    print("📝 logs/ - For log files")
    
    return directories

class MedVisionAI:
    def __init__(self, config_path='config/model_config.yaml'):
        # First, setup project directories
        setup_project_directories()
        
        # Setup logging
        self.setup_logging = setup_logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.loader = MedicalImageLoader()
        self.enhancer = MedicalImageEnhancer()
        self.feature_fusion = FeatureFusion()
        self.visualizer = ResultVisualizer()
        
        # Initialize models as None (will be set during training)
        self.cnn_model = None
        self.ml_models = None
        
        print("🚀 MedVision AI Agent Initialized!")
    
    def load_config(self, config_path):
        """Load configuration with fallback to default"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️ Config file {config_path} not found, creating default...")
            return self.create_default_config(config_path)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self.create_default_config(config_path)
    
    def create_default_config(self, config_path):
        """Create default configuration file"""
        default_config = {
            'model': {
                'name': 'LightweightMedicalCNN',
                'input_shape': [224, 224, 3],
                'num_classes': 2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50
            },
            'training': {
                'use_augmentation': True,
                'validation_split': 0.2,
                'early_stopping_patience': 10
            },
            'features': {
                'texture': {
                    'glcm_distances': [1, 2, 3],
                    'glcm_angles': [0, 0.785, 1.57, 2.356]
                }
            }
        }
        
        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"📝 Default configuration created at {config_path}")
        return default_config
    
    def check_data_structure(self, data_dir):
        """Check if data directory has proper structure"""
        required_dirs = ['train', 'val']
        
        for dir_name in required_dirs:
            dir_path = os.path.join(data_dir, dir_name)
            if not os.path.exists(dir_path):
                print(f"⚠️ Warning: {dir_path} not found")
                print("📁 Expected structure:")
                print("   data_dir/")
                print("   ├── train/")
                print("   │   ├── class1/")
                print("   │   └── class2/")
                print("   └── val/")
                print("       ├── class1/")
                print("       └── class2/")
                return False
        
        print("✅ Data structure looks good!")
        return True
    
    def train_cnn_model(self, train_dir, val_dir):
        """آموزش مدل CNN از صفر"""
        print("📚 Training CNN model from scratch...")
        
        # Check if directories exist
        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")
        if not os.path.exists(val_dir):
            raise ValueError(f"Validation directory not found: {val_dir}")
        
        # Create data generators
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=self.config['model']['batch_size'],
            class_mode='binary'
        )
        
        val_generator = train_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),
            batch_size=self.config['model']['batch_size'],
            class_mode='binary'
        )
        
        # Create and compile model
        model = LightweightMedicalCNN(
            num_classes=self.config['model']['num_classes']
        )
        
        trainer = MedicalModelTrainer(model, "lightweight_cnn")
        trainer.compile_model(learning_rate=self.config['model']['learning_rate'])
        
        # Train model
        history = trainer.train(train_generator, val_generator, epochs=self.config['model']['epochs'])
        
        # Evaluate
        results = trainer.evaluate(val_generator)
        
        # Save model
        model_path = "output/models/lightweight_cnn.h5"
        model.save(model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Visualize results
        viz_path = "output/visualizations/training_curves/training_history.png"
        self.visualizer.plot_training_history(history, viz_path)
        
        # Save training report
        self.save_training_report(results, history)
        
        # Set the trained model for later use
        self.cnn_model = model
        
        return model, results
    
    def save_training_report(self, results, history):
        """Save training results to report"""
        import json
        from datetime import datetime
        
        report = {
            'training_date': datetime.now().isoformat(),
            'model_name': 'LightweightMedicalCNN',
            'results': results,
            'final_accuracy': history.history['accuracy'][-1] if 'accuracy' in history.history else 0,
            'final_val_accuracy': history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0,
            'training_time': 'N/A'  # You can add timing logic
        }
        
        report_path = "output/reports/training/training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Training report saved to {report_path}")
    
    def extract_and_train_traditional(self, image_dir, label_file):
        """استخراج ویژگی‌ها و آموزش مدل‌های کلاسیک"""
        print("🔬 Extracting handcrafted features...")
        
        # استخراج ویژگی‌ها از تمام تصاویر
        features, labels = self.feature_fusion.extract_features_from_directory(
            image_dir, label_file
        )
        
        # آموزش مدل‌های کلاسیک
        ml_trainer = TraditionalMLModels()
        ml_trainer.train(features, labels)
        
        # Save traditional model
        import joblib
        model_path = "output/models/traditional_ml.joblib"
        joblib.dump(ml_trainer, model_path)
        print(f"💾 Traditional model saved to {model_path}")
        
        self.ml_models = ml_trainer
        return ml_trainer
    
    def predict_single_image(self, image_path, model_type='cnn'):
        """پیش‌بینی روی یک تصویر"""
        print(f"🔍 Analyzing image: {image_path}")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        # بارگذاری و پیش‌پردازش
        image = self.loader.load_image(image_path)
        enhanced = self.enhancer.enhance_contrast(image)
        
        if model_type == 'cnn':
            if self.cnn_model is None:
                # Try to load pre-trained model
                model_path = "output/models/lightweight_cnn.h5"
                if os.path.exists(model_path):
                    self.cnn_model = tf.keras.models.load_model(model_path)
                    print("✅ Loaded pre-trained CNN model")
                else:
                    raise ValueError("No trained CNN model found. Please train first.")
            
            # استفاده از CNN
            processed = self.enhancer.preprocess(enhanced)
            prediction = self.cnn_model.predict(np.expand_dims(processed, axis=0))
            
            # Save prediction
            self.save_prediction_result(image_path, prediction[0], model_type)
            
            return prediction[0]
        else:
            if self.ml_models is None:
                # Try to load pre-trained traditional model
                import joblib
                model_path = "output/models/traditional_ml.joblib"
                if os.path.exists(model_path):
                    self.ml_models = joblib.load(model_path)
                    print("✅ Loaded pre-trained traditional model")
                else:
                    raise ValueError("No trained traditional model found. Please train first.")
            
            # استفاده از ویژگی‌های دست‌ساز
            features = self.feature_fusion.extract_single_image_features(enhanced)
            prediction = self.ml_models.predict([features], model_type)
            
            # Save prediction
            self.save_prediction_result(image_path, prediction[0], model_type)
            
            return prediction[0]
    
    def save_prediction_result(self, image_path, prediction, model_type):
        """Save prediction result to file"""
        import json
        from datetime import datetime
        
        result = {
            'image_path': image_path,
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'predicted_class': np.argmax(prediction) if isinstance(prediction, (list, np.ndarray)) else prediction
        }
        
        # Create filename from image path
        image_name = os.path.basename(image_path).split('.')[0]
        result_path = f"output/predictions/single/{image_name}_prediction.json"
        
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 Prediction saved to {result_path}")
    
    def demo_mode(self):
        """Demo mode to show directory structure and basic functionality"""
        print("\n🎯 DEMO MODE: Project Structure")
        print("=" * 50)
        
        # Show created directories
        root_dirs = ['data', 'output', 'logs', 'config']
        for dir_name in root_dirs:
            if os.path.exists(dir_name):
                print(f"📁 {dir_name}/")
                # Show subdirectories
                for root, dirs, files in os.walk(dir_name):
                    level = root.replace(dir_name, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f'{indent}├── {os.path.basename(root)}/')
                    sub_indent = ' ' * 2 * (level + 1)
                    for file in files[:3]:  # Show first 3 files
                        print(f'{sub_indent}├── {file}')
                    if len(files) > 3:
                        print(f'{sub_indent}└── ... ({len(files) - 3} more files)')
                    break  # Only show first level for demo
        
        print("\n✅ Project is ready! Next steps:")
        print("1. Place your medical images in data/raw/")
        print("2. Run: python main.py --mode train --data_dir ./data/raw")
        print("3. Run: python main.py --mode predict --image_path ./data/raw/sample.jpg")

def main():
    parser = argparse.ArgumentParser(description='MedVision AI Agent')
    parser.add_argument('--mode', choices=['train', 'predict', 'serve', 'demo'], required=True,
                       help='Operation mode: train models, predict on image, serve API, or demo')
    parser.add_argument('--data_dir', help='Path to training data directory')
    parser.add_argument('--image_path', help='Path to image for prediction')
    parser.add_argument('--model_type', choices=['cnn', 'traditional'], default='cnn',
                       help='Model type to use for prediction')
    
    args = parser.parse_args()
    
    # Initialize agent (this will create directories automatically)
    agent = MedVisionAI()
    
    try:
        if args.mode == 'train':
            if not args.data_dir:
                raise ValueError("--data_dir is required for training mode")
            
            # Check data structure
            if not agent.check_data_structure(args.data_dir):
                print("❌ Please organize your data in the expected structure")
                return
            
            print("🏋️ Starting training process...")
            train_dir = os.path.join(args.data_dir, 'train')
            val_dir = os.path.join(args.data_dir, 'val')
            
            cnn_model, results = agent.train_cnn_model(train_dir, val_dir)
            
            print("✅ Training completed!")
            print(f"📊 Results: {results}")
        
        elif args.mode == 'predict':
            if not args.image_path:
                raise ValueError("--image_path is required for predict mode")
            
            prediction = agent.predict_single_image(args.image_path, args.model_type)
            print(f"🎯 Prediction: {prediction}")
        
        elif args.mode == 'serve':
            from src.api.streamlit_app import start_web_interface
            start_web_interface()
        
        elif args.mode == 'demo':
            agent.demo_mode()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
