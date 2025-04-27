import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from config import BenchmarkConfig

def load_and_preprocess_data():
    """Load CIFAR-10 and convert to tf.data.Dataset"""
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    #Normalize
    X_train = X_train.astype('float32') / 255 #the data in X_train might be in a different format, such as integers (int), which are not suitable for certain operations like normalization or training machine learning models.so we convert it to fractional values
    X_test = X_test.astype('float32') / 255

    # One-hot encode labels
    y_train = to_categorical(y_train, BenchmarkConfig.NUM_CLASSES) # to_categorical used to convert integer class labels into a one-hot encoded format.
    y_test = to_categorical(y_test, BenchmarkConfig.NUM_CLASSES)

    # Convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


    # Apply performance Optimization
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BenchmarkConfig.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BenchmarkConfig.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


