import os

class BenchmarkConfig:
    # Training Parameters
    BATCH_SIZE = 256
    EPOCHS = 5
    NUM_CLASSES = 10  # a parameter of the model, defining how many outputs the classifier has.


    # Path inside the container (mounted to our scratch folder)
    HOST_SCRATCH = "/mnt"
    LOG_DIR = os.path.join(HOST_SCRATCH,"logs")
    SAVE_DIR = os.path.join(HOST_SCRATCH,"saved_models")
    CPU_LOG_DIR = os.path.join(LOG_DIR, "cpu")
    GPU_LOG_DIR = os.path.join(LOG_DIR, "gpu")


    # Model architecture
    INPUT_SHAPE = (32 ,32 , 3)


    @classmethod
    def setup_dirs(cls):
        """Directory will be created on host, not in container"""
        pass


