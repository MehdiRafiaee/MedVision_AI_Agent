import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class MedicalAutoencoder(Model):
    def __init__(self, input_shape: tuple = (128, 128, 3), latent_dim: int = 256):
        super().__init__()
        self.input_shape_ = input_shape
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(8 * 8 * 256, activation='relu'),
            layers.Reshape((8, 8, 256)),
            layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(3, 3, strides=2, activation='sigmoid', padding='same')
        ])
    
    def call(self, inputs, training=False):
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
    
    def encode(self, inputs):
        """Encode inputs to latent space"""
        return self.encoder(inputs)
    
    def decode(self, latent_vectors):
        """Decode latent vectors to images"""
        return self.decoder(latent_vectors)
    
    def detect_anomaly(self, test_image, threshold: float = 0.1):
        """Detect anomalies based on reconstruction error"""
        # Reconstruct image
        reconstruction = self.call(tf.expand_dims(test_image, 0))
        
        # Calculate reconstruction error
        reconstruction_error = tf.reduce_mean(
            tf.square(test_image - reconstruction[0])
        ).numpy()
        
        # Classify as anomaly if error exceeds threshold
        is_anomaly = reconstruction_error > threshold
        
        return {
            'is_anomaly': is_anomaly,
            'reconstruction_error': reconstruction_error,
            'reconstruction': reconstruction[0],
            'threshold': threshold
        }
