import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from utils import load_data
from config import Config



def plot_sample(X,y, index):
    """Plot a single image with its true label"""
    plt.figure(figsize=(15,2))
    plt.imshow(X[index])
    plt.xlabel(Config.CLASSES[y[index][0]]) # CIFA-10 labels are 2D (e.g: [[3]])
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Generate predictions and classification report"""
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)


    # Flatten y_test if needed (original shape:(10000, 1))
    y_test_flat = y_test.reshape(-1)


    print("\nClassification Report:")
    print(classification_report(y_test_flat, y_pred_classes, target_names=Config.CLASSES))

    return y_pred




def evaluate():
    # Load model and data
    model = load_model(Config.SAVE_MODEL_PATH)
    (_,_), (X_test, y_test) = load_data()

    # 1. Basic Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)  # testing the model with test dataset
    printf(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")


    # 2. Plot sample images
    print("\n Sample Test Images:")
    for i in range(3):
        plot_sample(X_test, y_test, i)

    # 3. Generate predictions and classification report
    y_pred = evaluate_model(model, X_test, y_test)

    # 4. Show predicted class for sample
    sample_idx = 1
    y_pred_class = np.argmax(y_pred[sample_idx])
    print(f"\nPrediction for sample {sample_idx}:")
    print(f"predicted: {Config.CLASSES[y_pred_class]}")
    print(f"Actual: {Config.CLASSES[y_test[sample_idx][0]]}")


if __name__ == "__main__":
    evaluate()

