"""
Model Architecture Module
ResNet50 with transfer learning for pothole detection
"""

from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_model(input_shape, backbone='resnet50', trainable_layers=10):
    """
    Build a CNN model using transfer learning for binary classification.
    
    Args:
        input_shape: Tuple (height, width, channels)
        backbone: 'resnet50', 'efficientnet', or 'mobilenet'
        trainable_layers: Number of layers from the top to make trainable
    
    Returns:
        Compiled Keras model
    """
    try:
        # Select backbone
        if backbone == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        elif backbone == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        elif backbone == 'mobilenet':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze base model initially
        for layer in base_model.layers:
            layer.trainable = False
        
        # Unfreeze top layers for fine-tuning
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        
        # Build classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        
        print(f"\n=== Model Architecture ({backbone}) ===")
        print(f"Total params: {model.count_params():,}")
        print(f"Trainable params: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")
        print(f"Non-trainable params: {sum([np.prod(v.shape) for v in model.non_trainable_weights]):,}")
        
        return model
        
    except Exception as e:
        print(f"Error building model: {e}")
        raise


import numpy as np