import sys
import os
import pytest
import numpy as np
import tensorflow as tf

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model import create_model

def test_model_creation():
    """Test if the model can be created with different input shapes and number of classes."""
    input_shape = (187, 1)
    num_classes = 5
    
    model = create_model(input_shape, num_classes)
    
    # Test model output shape
    test_input = np.random.random((1, *input_shape))
    output = model.predict(test_input)
    
    assert output.shape == (1, num_classes)
    assert np.allclose(np.sum(output, axis=1), 1.0)  # Check if probabilities sum to 1

def test_model_compilation():
    """Test if the model is properly compiled with the correct loss and metrics."""
    input_shape = (187, 1)
    num_classes = 5
    
    model = create_model(input_shape, num_classes)
    
    assert model.loss == 'sparse_categorical_crossentropy'
    assert 'accuracy' in [metric.name for metric in model.metrics]

if __name__ == "__main__":
    pytest.main([__file__]) 