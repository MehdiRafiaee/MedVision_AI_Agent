# src/models/lightweight_cnn.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class LightweightMedicalCNN(Model):
    def __init__(self, num_classes=2, input_shape=(224, 224, 3)):
        super().__init__()
        self.num_classes = num_classes
        
        # Block 1
        self.conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2)
        self.drop1 = layers.Dropout(0.2)
        
        # Block 2
        self.conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2)
        self.drop2 = layers.Dropout(0.3)
        
        # Block 3
        self.conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(2)
        self.drop3 = layers.Dropout(0.4)
        
        # Block 4
        self.conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.bn4 = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D(2)
        self.drop4 = layers.Dropout(0.5)
        
        # Classification Head
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(128, activation='relu')
        self.drop5 = layers.Dropout(0.5)
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.drop1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.drop2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        x = self.drop3(x, training=training)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.pool4(x)
        x = self.drop4(x, training=training)
        
        x = self.global_pool(x)
        x = self.dense1(x)
        x = self.drop5(x, training=training)
        
        return self.classifier(x)
    
    def build_graph(self):
        """برای نمایش summary"""
        x = tf.keras.Input(shape=(224, 224, 3))
        return Model(inputs=[x], outputs=self.call(x))
