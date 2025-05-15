import tensorflow as tf
from tensorflow.keras import layers, Model

def create_model(input_shape, num_classes):
    """
    Create a CNN-LSTM model for ECG classification.
    
    Args:
        input_shape (tuple): Shape of input data (time_steps, features)
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: Compiled model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Example usage
    input_shape = (187, 1)  # Example shape for ECG data
    num_classes = 5  # Example number of classes
    model = create_model(input_shape, num_classes)
    model.summary() 