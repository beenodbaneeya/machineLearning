from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from config import BenchmarkConfig

def create_simple_nn():
    """
    Create a simple neural network model
    Architecture:
    - Input : Flattened 32 * 32 * 3 CIFAR-10 images
    - Hidden : 3000 neurons (ReLU)
    - Hidden : 1000 neurons (ReLU)
    - Output : 10 neurons (Sigmoid)
    

    note: this part of code is written here to demonstrate on how we used to pass the input layer inside Flatten
    .If you do this you will get the error/ warning. Recommended way of doing is given below
    model = models.Sequential([
        layers.Flatten(shape=BenchmarkConfig.INPUT_SHAPE),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(BenchmarkConfig.NUM_CLASSES, activation='sigmoid')

    ])

    """

    model = models.Sequential([
        layers.Input(input_shape=BenchmarkConfig.INPUT_SHAPE),
        layers.Flatten(),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(BenchmarkConfig.NUM_CLASSES, activation='sigmoid')

    ])
  

    model.compile(
        optimizer=optimizers.SGD(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Created simple neural network (3000-1000-10 architecture)")
    return model