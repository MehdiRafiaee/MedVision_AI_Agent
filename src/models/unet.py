import tensorflow as tf
from tensorflow.keras import layers, Model

class DoubleConv(layers.Layer):
    def __init__(self, filters: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        return x

class UpConv(layers.Layer):
    def __init__(self, filters: int, kernel_size: int = 2):
        super().__init__()
        self.up = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same')
        self.concat = layers.Concatenate()
    
    def call(self, inputs, skip_connection):
        x = self.up(inputs)
        x = self.concat([x, skip_connection])
        return x

class UNet(Model):
    def __init__(self, input_shape: tuple = (256, 256, 3), num_classes: int = 1):
        super().__init__()
        self.input_shape_ = input_shape
        self.num_classes = num_classes
        
        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(64)
        self.pool1 = layers.MaxPool2D(2)
        
        self.enc2 = DoubleConv(128)
        self.pool2 = layers.MaxPool2D(2)
        
        self.enc3 = DoubleConv(256)
        self.pool3 = layers.MaxPool2D(2)
        
        self.enc4 = DoubleConv(512)
        self.pool4 = layers.MaxPool2D(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(1024)
        
        # Decoder (Expansive Path)
        self.up1 = UpConv(512)
        self.dec1 = DoubleConv(512)
        
        self.up2 = UpConv(256)
        self.dec2 = DoubleConv(256)
        
        self.up3 = UpConv(128)
        self.dec3 = DoubleConv(128)
        
        self.up4 = UpConv(64)
        self.dec4 = DoubleConv(64)
        
        # Output layer
        self.output_conv = layers.Conv2D(num_classes, 1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # Encoder
        e1 = self.enc1(inputs, training=training)           # (256,256,64)
        e2 = self.enc2(self.pool1(e1), training=training)   # (128,128,128)
        e3 = self.enc3(self.pool2(e2), training=training)   # (64,64,256)
        e4 = self.enc4(self.pool3(e3), training=training)   # (32,32,512)
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4), training=training)  # (16,16,1024)
        
        # Decoder with skip connections
        d1 = self.up1(b, e4)
        d1 = self.dec1(d1, training=training)               # (32,32,512)
        
        d2 = self.up2(d1, e3)
        d2 = self.dec2(d2, training=training)               # (64,64,256)
        
        d3 = self.up3(d2, e2)
        d3 = self.dec3(d3, training=training)               # (128,128,128)
        
        d4 = self.up4(d3, e1)
        d4 = self.dec4(d4, training=training)               # (256,256,64)
        
        # Output
        outputs = self.output_conv(d4)                      # (256,256,num_classes)
        
        return outputs
    
    def build_graph(self):
        """Build model graph for summary"""
        x = tf.keras.Input(shape=self.input_shape_)
        return Model(inputs=[x], outputs=self.call(x))
