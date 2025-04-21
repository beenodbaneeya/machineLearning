import tensorflow as tf
from model import create_cnn_model
from utils import load_data
from config import Config
import os
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

# setup logging
def setup_tensorboard_callback():
    #Cluster-safe TensorBoard callback (only writes logs, doesn't start server)
    log_dir = os.path.join(
        Config.LOG_DIR,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(log_dir, exist_ok=True)

    return TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,      # Log weight histograms
        write_graph=True,      # Log model architecture
        write_images=False,     # Disable image logging (saves space)
        update_freq='epoch',    # Log at epoch end
        profile_batch=0        # Disable profiling (cluster-safe)
    )

def plot_training_history(history):
    #Plot training/validation accuracy and loss#
    plt.figure(figsize=(12,4))

    #Accuracy Plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


    #Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train():
    #Load data and model
    (X_train,y_train), (X_test, y_test) = load_data()
    model = create_cnn_model()
    callbacks = [setup_tensorboard_callback()]  # Use the configured callback


    # Train
    history = model.fit(
        X_train, y_train,
        batch_size = Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    plot_training_history(history)


    # Save Model
    os.makedirs(os.path.dirname(Config.SAVE_MODEL_PATH), exist_ok=True)
    model.save(Config.SAVE_MODEL_PATH)
    print(f"Training data shape: {X_train.shape}")
    print(f"Model saved to {Config.SAVE_MODEL_PATH}")


if __name__ == "__main__":
    train()

