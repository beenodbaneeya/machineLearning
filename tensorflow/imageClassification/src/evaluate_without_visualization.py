"""
CIFAR-10 CNN Model Evaluation 
----------------------------------------------
Text-based evaluation of CIFAR-10 model performance.
"""

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from utils import load_data
from config import Config

def print_sample_info(X_test, y_test, index):
    """
    Text-based alternative to image visualization
    Prints image metadata and label information
    
    Args:
        X_test (numpy.ndarray): Test images (10000, 32, 32, 3)
        y_test (numpy.ndarray): Test labels (10000, 1)
        index (int): Sample index to describe
    """
    sample = X_test[index]
    print(f"\nSample {index}:")
    print(f"- Shape: {sample.shape}")
    print(f"- Min/Max/Mean values: {sample.min():.1f}/{sample.max():.1f}/{sample.mean():.1f}")
    print(f"- True Label: {Config.CLASSES[y_test[index][0]]}")

def evaluate_model(model, X_test, y_test):
    """
    Generate predictions and classification report
    
    Args:
        model (tf.keras.Model): Trained CNN model
        X_test (numpy.ndarray): Test images
        y_test (numpy.ndarray): Test labels
    
    Returns:
        numpy.ndarray: Model predictions (10000, 10)
    """
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_flat = y_test.reshape(-1)
    
    print("\nClassification Report:")
    print(classification_report(
        y_test_flat, 
        y_pred_classes, 
        target_names=Config.CLASSES
    ))
    
    return y_pred

def evaluate():
    """Main evaluation pipeline without visualization"""
    model = load_model(Config.SAVE_MODEL_PATH)
    (_, _), (X_test, y_test) = load_data()
    
    # Basic evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    # Text-based sample information
    print("\nSample Descriptions:")
    for i in range(3):  # First 3 samples
        print_sample_info(X_test, y_test, i)
    
    # Predictions analysis
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Example prediction
    sample_idx = 1
    y_pred_class = np.argmax(y_pred[sample_idx])
    print(f"\nPrediction Example:")
    print(f"Sample {sample_idx}:")
    print(f"- Predicted: {Config.CLASSES[y_pred_class]}")
    print(f"- Actual: {Config.CLASSES[y_test[sample_idx][0]]}")

if __name__ == "__main__":
    evaluate()