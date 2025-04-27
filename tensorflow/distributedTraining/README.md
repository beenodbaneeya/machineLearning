# Utilize tensorflow mirror strategy to perform the distributed training on the multi-gpu system
We can use the four gpus to do the image classification for small image dataset.We have total of 50k datasets and we will be building a convolutional neural network and we will split all these images in such a way that we will take a 1k image batch and we will distribute those 1k to 4 GPUs and then perform the distributed training.This is useful when we have the huge dataset . for instance million images or humungous database. Distributed training helps to get better performance. So this tutorial will guide you on how to distribute the training samples and distributing it to the numbers of gps we have.

## One-hot encoding example
One-hot encoding is used because many machine learning models, especially neural networks, require the target labels (y_train, y_test) to be in a specific format.
It ensures:
1. compability to loss function .i.e. if the true label is 2 and the predicted probabilities are [0.1, 0.3, 0.6], the one-hot encoded label [0, 0, 1] is used to compute the loss.
2. multi -class classification : if the true label is 2 and the predicted probabilities are [0.1, 0.3, 0.6], the one-hot encoded label [0, 0, 1] is used to compute the loss.
3. Avoiding Ambuiguity: If the labels are integers (e.g., 0, 1, 2), the model might mistakenly interpret them as continuous values. One-hot encoding eliminates this ambiguity by representing each class as a distinct vector.

Example: 
Before One-Hot Encoding:
Suppose y_train contains the following class labels:
````bash
y_train = [0, 1, 2, 1]
````
After One-Hot Encoding:
If NUM_CLASSES = 3, the one-hot encoded version will be:
````bash
y_train = [
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [0, 0, 1],  # Class 2
    [0, 1, 0]   # Class 1
]
````

## tf.data.Dataset
We have to convert the numpy arrays into tensorlfow dataset so that it could support batching.We require data pipeline becasue sometimes we might have extensively large dataset and hence we want to read the data from the disk step by step. for intance if the size of the database is 2 terabyte and if we have 32 gigabyte memory, it might not fit if we want to read everything at once.So this tensorflow datasets helps us to build the pipeline where we can gradually read our data step by step to improve the efficiency of our code.
The following performance optimizations can be included in the Dataset API.
1. shuffle(): Better training dynamics
2. prefetch(): Overlaps data preprocessing and model execution
3. AUTOTUNE: Lets TensorFlow  optimize the buffer sizes

This approach helps us to move batching into dataset pipeline.This will helps us get the performance benefits i.e 2-3x faster data loading, better memory efficiency and it is essential for multi-GPU training

## warm-up phase
Warm-up refers to running a few initial training steps before actual training logic.It is beneficial in following ways:
1. Initialize GPU state - CUDA/ROCm kernels compile on first run
2. Load data into cache - Pre-fill memory buffers
3. Stabilize clocks - Modern GPUs boost clocks gradually

First epoch is often slower due to one-time overheads. so it is helpful for optimization.

## Difference between  DDP (PyTorch) and MirroredStrategy (TensorFlow)

For DDP:

1 Python process per GPU	
Tensor is created locally on each GPU	
Communication handled automatically after tensor creation	Need to run computations inside strategy.run() to sync replicas

For MirroredStrategy (TensorFlow)
1 Python process sees all GPUs
Tensor must be replicated via strategy.run()
Need to run computations inside strategy.run() to sync replicas

## if needed to verify GPU first include this code in the job script
````bash
### GPU VERIFICATION TEST ###
echo "=== Running GPU Verification Test ==="

srun --export=ALL singularity exec --bind $HOST_SCRATCH:/mnt $CONTAINER \
  bash -c "
    source /opt/miniconda3/bin/activate tensorflow
    python gpu_verify.py
  "
````

## CPU benchmarking not calculated with this code
Note that, since we utilize the GPU for running the code, we were not able to get the speedup. But we can further continue this work to get the actual speedup too.