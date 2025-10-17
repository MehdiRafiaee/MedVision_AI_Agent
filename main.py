# main.py
#!/usr/bin/env python3
"""
MedVision AI Agent - Main Entry Point
Complete from-scratch medical image analysis system
"""

import os
import argparse
import yaml
import numpy as np

from src.preprocessor.image_loader import MedicalImageLoader
from src.preprocessor.enhancement import MedicalImageEnhancer
from src.features.feature_fusion import FeatureFusion
from src.models.lightweight_cnn import LightweightMedicalCNN
from src.models.traditional_ml import TraditionalMLModels
from src.trainer.model_trainer import MedicalModelTrainer
from src.evaluator.visualization import ResultVisualizer
from src.utils.logger import setup_logging

class MedVisionAI:
    def __init__(self, config_path='config/model_config.yaml'):
        self.setup_logging = setup_logging
        self.setup_logging()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.loader = MedicalImageLoader()
        self.enhancer = MedicalImageEnhancer()
        self.feature_fusion = FeatureFusion()
        self.visualizer = ResultVisualizer()
        
        print("🚀 MedVision AI Agent Initialized!")
    
    def train_cnn_model(self, train_dir, val_dir):
        """آموزش مدل CNN از صفر"""
        print("📚 Training CNN model from scratch...")
        
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
        
        # Visualize results
        self.visualizer.plot_training_history(history, 'output/training_history.png')
        
        return model, results
    
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
        
        return ml_trainer
    
    def predict_single_image(self, image_path, model_type='cnn'):
        """پیش‌بینی روی یک تصویر"""
        print(f"🔍 Analyzing image: {image_path}")
        
        # بارگذاری و پیش‌پردازش
        image = self.loader.load_image(image_path)
        enhanced = self.enhancer.enhance_contrast(image)
        
        if model_type == 'cnn':
            # استفاده از CNN
            processed = self.enhancer.preprocess(enhanced)
            prediction = self.cnn_model.predict(np.expand_dims(processed, axis=0))
            return prediction[0]
        else:
            # استفاده از ویژگی‌های دست‌ساز
            features = self.feature_fusion.extract_single_image_features(enhanced)
            prediction = self.ml_models.predict([features], model_type)
            return prediction[0]

def main():
    parser = argparse.ArgumentParser(description='MedVision AI Agent')
    parser.add_argument('--mode', choices=['train', 'predict', 'serve'], required=True)
    parser.add_argument('--data_dir', help='Path to training data')
    parser.add_argument('--image_path', help='Path to image for prediction')
    parser.add_argument('--model_type', choices=['cnn', 'traditional'], default='cnn')
    
    args = parser.parse_args()
    
    agent = MedVisionAI()
    
    if args.mode == 'train':
        if not args.data_dir:
            raise ValueError("Data directory required for training")
        
        # تقسیم داده به train/validation
        # (در اینجا می‌توانید از train_test_split استفاده کنید)
        
        print("🏋️ Starting training process...")
        cnn_model, results = agent.train_cnn_model(
            os.path.join(args.data_dir, 'train'),
            os.path.join(args.data_dir, 'val')
        )
        
        print("✅ Training completed!")
        print(f"📊 Results: {results}")
    
    elif args.mode == 'predict':
        if not args.image_path:
            raise ValueError("Image path required for prediction")
        
        prediction = agent.predict_single_image(args.image_path, args.model_type)
        print(f"🎯 Prediction: {prediction}")
    
    elif args.mode == 'serve':
        from src.api.streamlit_app import start_web_interface
        start_web_interface()

if __name__ == "__main__":
    main()
