import tensorflow as tf
from model import create_cnn_model
from utils import load_data
from config import Config
import os
import datetime
import json  # Added for saving training history
from tensorflow.keras.callbacks import TensorBoard

def setup_tensorboard_callback():
    """Cluster-safe TensorBoard callback (only writes logs)"""
    log_dir = os.path.join(
        Config.LOG_DIR,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    
    return TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=0
    )

def save_training_history(history):
    """
    Save training metrics to JSON file instead of plotting
    Args:
        history: Keras History object containing training metrics
    """
    history_path = os.path.join(Config.LOG_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"Training metrics saved to {history_path}")

def print_training_summary(history):
    """Print key training metrics to console"""
    print("\nTraining Summary:")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

def train():
    # Load data and model
    (X_train, y_train), (X_test, y_test) = load_data()
    model = create_cnn_model()
    callbacks = [setup_tensorboard_callback()]

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2  # More detailed progress output
    )

    # Save results
    save_training_history(history)
    print_training_summary(history)
    
    os.makedirs(os.path.dirname(Config.SAVE_MODEL_PATH), exist_ok=True)
    model.save(Config.SAVE_MODEL_PATH)
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Model saved to {Config.SAVE_MODEL_PATH}")

if __name__ == "__main__":
    train()