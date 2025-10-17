# src/trainer/model_trainer.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class MedicalModelTrainer:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.history = None
        
        # Callbacks
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                f'models/{model_name}_best.h5', save_best_only=True, monitor='val_loss'
            )
        ]
    
    def compile_model(self, learning_rate=0.001):
        """کامپایل مدل"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def train(self, train_data, val_data, epochs=100):
        """آموزش مدل"""
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=1
        )
        return self.history
    
    def evaluate(self, test_data):
        """ارزیابی مدل"""
        results = self.model.evaluate(test_data, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        
        # پیش‌بینی برای گزارش دقیق‌تر
        y_pred = np.argmax(self.model.predict(test_data), axis=1)
        y_true = np.concatenate([y for x, y in test_data], axis=0)
        
        # محاسبه ماتریس درهم‌ریختگی
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report
        }
