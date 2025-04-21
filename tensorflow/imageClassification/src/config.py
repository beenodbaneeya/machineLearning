import os

class Config:
    # Data
    NUM_CLASSES = 10
    INPUT_SHAPE = (32,32,3)
    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Training
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    #paths
    SCRATCH_ROOT="/scratch/project_number/binod/tensorflow/"
    SAVE_MODEL_PATH = os.path.join(SCRATCH_ROOT,"saved_models", "cifar10_cnn.h5")
    LOG_DIR = os.path.join(SCRATCH_ROOT, "logs")

config = Config()