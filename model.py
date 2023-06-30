# In this file, we will define the architecture for our Keras model (following the YOLOv1 paper)
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU


cnn_architecture_definition = [
    {'layer_type': 'Conv2D', 'filters': 64, 'kernel_size': 7, 'strides': 2, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'MaxPooling2D', 'pool_size': 2, 'strides': 2, 'padding': 'valid'},
    {'layer_type': 'Conv2D', 'filters': 192, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'MaxPooling2D', 'pool_size': 2, 'strides': 2, 'padding': 'valid'},
    {'layer_type': 'Conv2D', 'filters': 128, 'kernel_size': 1, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 256, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 256, 'kernel_size': 1, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 512, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'MaxPooling2D', 'pool_size': 2, 'strides': 2, 'padding': 'valid'},
    {'layer_type': 'Conv2D', 'filters': 256, 'kernel_size': 1, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 512, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 256, 'kernel_size': 1, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 512, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 256, 'kernel_size': 1, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 512, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 512, 'kernel_size': 1, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 1024, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'MaxPooling2D', 'pool_size': 2, 'strides': 2, 'padding': 'valid'},
    {'layer_type': 'Conv2D', 'filters': 512, 'kernel_size': 1, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 1024, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 512, 'kernel_size': 1, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 1024, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 1024, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 1024, 'kernel_size': 3, 'strides': 2, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 1024, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'},
    {'layer_type': 'Conv2D', 'filters': 1024, 'kernel_size': 3, 'strides': 1, 'padding': 'same'},
    {'layer_type': 'BatchNormalization'},
    {'layer_type': 'LeakyReLU'}
]

# Define the YOLOv1 model architecture based on the above definition
# First, we define the CNN layers
# Then, based on split_size, num_boxes and num_classes, we define the output layers
class YOLOv1(tf.keras.Model):
    def __init__(self, num_boxes, num_classes, split_size=7):
        self.num_classes = num_classes
        self.split_size = split_size
        self.num_boxes = num_boxes
        
    def _create_cnn_layers(self):
        model = tf.keras.models.Sequential()
        for layer in cnn_architecture_definition:
            if layer['layer_type'] == 'Conv2D':
                model.add(Conv2D(layer['filters'], layer['kernel_size'], strides=layer['strides'], padding=layer['padding']))
            elif layer['layer_type'] == 'BatchNormalization':
                model.add(tf.keras.layers.BatchNormalization())
            elif layer['layer_type'] == 'LeakyReLU':
                model.add(LeakyReLU(alpha=0.1))
            elif layer['layer_type'] == 'MaxPooling2D':
                model.add(MaxPooling2D(pool_size=layer['pool_size'], strides=layer['strides'], padding=layer['padding']))
        return model

    # Define the output layers, starting with a Flatten layer, then Dense, then Reshape
    def _create_output_layers(self):
        # Calculate the number of output filters based on the number of boxes and classes
        num_output_filters = self.num_classes + self.num_boxes * 5
        
        # Create the output layers
        model = tf.keras.models.Sequential()
        
        # Flatten the output from the CNN layers
        model.add(Flatten())
        
        # Add a Dense layer with 4096 units
        model.add(Dense(4096))
        model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.1))
        
        # Add a Dense layer with: split_size * split_size * num_output_filters units
        model.add(Dense(self.split_size * self.split_size * num_output_filters))

        return model
    
    # Define the forward pass
    def call(self, inputs):
        cnn_layers = self._create_cnn_layers()
        output_layers = self._create_output_layers()
        
        # Pass the inputs through the CNN layers
        x = cnn_layers(inputs)
        
        # Pass the output from the CNN layers through the output layers
        x = output_layers(x)
        
        # Reshape the output from the output layers into a 7x7x30 tensor
        x = tf.reshape(x, (-1, self.split_size, self.split_size, self.num_classes + self.num_boxes * 5))
        
        return x
