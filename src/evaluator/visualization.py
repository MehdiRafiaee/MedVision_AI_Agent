# src/evaluator/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2

class ResultVisualizer:
    def __init__(self):
        plt.style.use('default')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    def plot_training_history(self, history, save_path=None):
        """نمایش تاریخچه آموزش"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0,0].plot(history.history['loss'], label='Training Loss', color=self.colors[0])
        axes[0,0].plot(history.history['val_loss'], label='Validation Loss', color=self.colors[1])
        axes[0,0].set_title('Model Loss')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].legend()
        
        # Accuracy
        axes[0,1].plot(history.history['accuracy'], label='Training Accuracy', color=self.colors[0])
        axes[0,1].plot(history.history['val_accuracy'], label='Validation Accuracy', color=self.colors[1])
        axes[0,1].set_title('Model Accuracy')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].legend()
        
        # Precision
        axes[1,0].plot(history.history['precision'], label='Training Precision', color=self.colors[0])
        axes[1,0].plot(history.history['val_precision'], label='Validation Precision', color=self.colors[1])
        axes[1,0].set_title('Model Precision')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].legend()
        
        # Recall
        axes[1,1].plot(history.history['recall'], label='Training Recall', color=self.colors[0])
        axes[1,1].plot(history.history['val_recall'], label='Validation Recall', color=self.colors[1])
        axes[1,1].set_title('Model Recall')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attention_maps(self, image, attention_weights, save_path=None):
        """نمایش نقشه‌های توجه"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # تصویر اصلی
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # نقشه توجه
        attention_map = cv2.resize(attention_weights, (image.shape[1], image.shape[0]))
        im = axes[1].imshow(attention_map, cmap='hot')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # تلفیق
        blended = cv2.addWeighted(image, 0.7, 
                                cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET), 
                                0.3, 0)
        axes[2].imshow(blended)
        axes[2].set_title('Blended Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
