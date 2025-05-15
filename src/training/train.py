import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.preprocess import load_data, preprocess_data
from src.models.model import create_model

def train_model(
    data_path,
    input_shape,
    num_classes,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    model_save_path=None
):
    """
    Train the ECG classification model.
    
    Args:
        data_path (str): Path to the training data
        input_shape (tuple): Shape of input data
        num_classes (int): Number of output classes
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        validation_split (float): Proportion of data to use for validation
        model_save_path (str): Path to save the trained model
    """
    # Load and preprocess data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Reshape data for CNN-LSTM model
    X_train = X_train.reshape(X_train.shape[0], *input_shape)
    X_test = X_test.reshape(X_test.shape[0], *input_shape)
    
    # Create and compile model
    model = create_model(input_shape, num_classes)
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save model if path is provided
    if model_save_path:
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
    
    return model, history

if __name__ == "__main__":
    # Example usage
    data_path = "../../data/mitbih_train.csv"
    input_shape = (187, 1)  # Example shape for ECG data
    num_classes = 5  # Example number of classes
    
    # Create timestamp for model save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"../../models/ecg_model_{timestamp}"
    
    model, history = train_model(
        data_path=data_path,
        input_shape=input_shape,
        num_classes=num_classes,
        model_save_path=model_save_path
    ) 