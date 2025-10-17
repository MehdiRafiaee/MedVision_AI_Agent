# src/models/attention_network.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class AttentionBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
        self.conv_query = layers.Conv2D(filters, 1, activation='linear')
        self.conv_key = layers.Conv2D(filters, 1, activation='linear')
        self.conv_value = layers.Conv2D(filters, 1, activation='linear')
        self.conv_output = layers.Conv2D(filters, 1, activation='linear')
        self.gamma = self.add_weight(shape=(1,), initializer='zeros', trainable=True)
    
    def call(self, x):
        batch_size, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Projections
        query = self.conv_query(x)  # [B, H, W, C]
        key = self.conv_key(x)      # [B, H, W, C]
        value = self.conv_value(x)  # [B, H, W, C]
        
        # Reshape for attention
        query_flat = tf.reshape(query, [batch_size, h * w, self.filters])  # [B, N, C]
        key_flat = tf.reshape(key, [batch_size, h * w, self.filters])      # [B, N, C]
        value_flat = tf.reshape(value, [batch_size, h * w, self.filters])  # [B, N, C]
        
        # Attention scores
        attention_scores = tf.matmul(query_flat, key_flat, transpose_b=True)  # [B, N, N]
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)          # [B, N, N]
        
        # Apply attention
        attended = tf.matmul(attention_weights, value_flat)  # [B, N, C]
        attended = tf.reshape(attended, [batch_size, h, w, self.filters])
        
        # Residual connection
        output = self.conv_output(attended)
        return x + self.gamma * output

class AttentionMedicalCNN(Model):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Initial conv
        self.initial_conv = layers.Conv2D(32, 7, strides=2, padding='same', activation='relu')
        self.initial_bn = layers.BatchNormalization()
        self.initial_pool = layers.MaxPooling2D(3, strides=2, padding='same')
        
        # Attention blocks
        self.attention1 = AttentionBlock(32)
        self.conv1 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')
        
        self.attention2 = AttentionBlock(64)
        self.conv2 = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')
        
        # Classification
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.5)
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_pool(x)
        
        x = self.attention1(x)
        x = self.conv1(x)
        
        x = self.attention2(x)
        x = self.conv2(x)
        
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        
        return self.classifier(x)
