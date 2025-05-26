# Understanding the code
In this section we will try to understand the code written in the python files so that it will be more clear for us to get the insights on the model training part and the Distributed Data Parallel implementation part in details.

## dataloader.py
This module provides a utility function to create a PyTorch DataLoader for the MNIST dataset. The MNIST [MNIST Doc PyTorch](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST) dataset in pyTorch is a built-in dataset that provides a collection of handwritten digits(0-9) commonly used for training and testing machine learning models, particularly in computer vision tasks. It consists of 60,000 training images and 10,000 testing images, each of size 28*28  pixels in grayscale.

The DataLoader is configured to preprocess the data and load it in batches, making it ready for training machine learning models.

`from torchvision import datasets, transforms` This line of code imports datasets and transforms from torchvision. The `datasets` module provides access to the popular datasets including MNIST. You can learn more about the available datasets here [datasets](https://docs.pytorch.org/vision/stable/datasets.html). The `transforms` module contains tools for data preprocessing and augmentation.Transforms can be used to transform or augment data for training or inference of different tasks (image classification, detection, segmentation, video classification).We can learn more about various transformation we can apply to our data here. [transform](https://docs.pytorch.org/vision/stable/transforms.html)

Moving forward, this line of code `from torch.utils.data import DataLoader` import DataLoader which is a PyTorch utility to load data in batches, shuffle it and handle multiprocessing for efficiency.To learn more about the DataLoader, please visit this link.[DataLoader](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

We define `get_mnist_dataloader` function which returns a PyTorch DataLoader object that provides an iterable over the MNIST training dataset.Here, to demonstrate the bottleneck issue,we pass default `batch_size=32` as the first argument which are the number of samples to load per patch and `num_workers=4` which is the number of subprocesses to use for the data loading. 

Then we apply the transformation using `transforms.Compose`  which combines multiple transformations into a single pipeline.The first transformation in our case is `transforms.ToTensor` which converts a PIL image or NumPy array into a PyTorch tensor and scales pixel valuee to range between [0,1]. Then the second transformation is `transforms.Normalize` which normalizes the tensor using the specified mean and standard deviation.In our case the values that we use there re specific to the MNIST dataset. After the transformation, we are loading the MNIST dataset using `datasets.MNIST` where we pass different parameters.One of the parameter here will be the transformation that we applied earlier. For more details about is parameters, please visit this page [MNIST dataset object](https://docs.pytorch.org/vision/stable/datasets.html#mnist). Hence,DataLoader object is created which wraps the datasets to provide an iterable for loading data in batches.




## model.py

 The `SimpleModel` class defines a fully connected feedforward neural network with two layers. It is created to classify flattened 28*28 grayscale images (e.g. MNIST) into 10 categories.

 The first module we import here is `torch` which is the core PyTorch library for tensor operations and deep learning using GPUs and CPUs.The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors.To read more about the torch , please visit this link.[torch](https://docs.pytorch.org/docs/stable/torch.html)
The second module we import is the `torch.nn` which contains building blocks for neural networks, such as layes, activation fucntion and loss functions.To read more about the available layers and utilities, check this page [torch.nn](https://docs.pytorch.org/docs/stable/nn.html)

We we define the class, we inherit all the base class for all neural netwrok models in PyTorch.Then we have a constructor which initializes the model´s layers, where `self.fc1` is a fully connected linear layers with input size `28*28`which corresponds to the flattened image pixels and output size is `128` which is the number of neurons in the first hidden layer. We then have, another fully connected linear layer with input size of `128` which is the output from the first layer and output size of `10` which corresponds to the 10 output classes i.e. digits `0-9`.

Our `SimpleModel` class contains a method `forward` which defines the forward pass of the model. i.e. how the input data flows through the layers to produce the output.First, we flattened the input where the input tensor `x` is reshaped to have a shape of `[batch_szie, 28*28]`. And in our case, `-1` infers to the batch size. This step is crucial because the fully connected layers expect a 2D input of shape `[batch_szie, num_features]`. Then we pass that flattened input through the first fully connected layer `fc1` togethar with applying the Rectified Linear Unit(ReLU) activation function to introduce non-linearity. To learn about this ReLU activation function, refer to this page [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)

Then we pass the output of the first layer through the second fully connected layer`fc2`. The output is a tensor of shape `batch_size, 10` which represents the raw scores i.e. logits for each of the 10 classes. The term `logits` refer to the unnormalized predictions(output values) generate by the finaø layer of the neural network before applying a Softmax or sigmoid activation function.Then later Softmax converts this logits to probabiities (sum to 1). The PyTorch´s `CrossEntropyLoss` expects logits and not probabilites so we return this from our model class.



## single_gpu_train.py

This script defines the training process for the `SimpleModel` neural network using the MNIST dataset. It includes functionality for model training, loss computation , optimization and profiling to analyze performance.

The modules that were imporeted in this script are discussed below:
1. `torch.nn` : As discussed earlier in the `model.py` part , this script import this module to define the loss function i.e. `CrossEntropyLoss`
2. `torch.optim` : This module gives aceess to the optimization algorithms such as Adam. To learn more about Adam optimizer please refer to this link [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
3. `torch.profiler`: This is a utility for profiling PyTorch code to analyze performance and resource usage.To learn more about the PyTorch Profiler , please visit this link [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

 Along with these modules we have also imported the ones that we have created. i.e. model and dataloader.

 Then we define the train function that handles the entire training process for the `SimpleModel`. It takes epochs, batch_size and num_workers as parameters. A single epoch is the full passes through the dataser to train the model and in our case we set it to 5 by default for the simplicity. We set the `batch_szie` to 32 which is the number of samples per batch. And finally, `num_workers` is the number of subprocess used for data loading which we set as 4 for the testing purposes.
 
 
 First the model is intialized and moved to the appropirate device (CPU or GPU). It selects GPU (CUDA) if available otherwise defaults to CPU. It is a good practice to verify that your system has CUDA avalable by using this code `torch.cuda.is_available()` before running your training.Then we setup the loss function and optimizer followed by loading the MNIST dataset using the DataLoader. After that we iterate through epochs and batches to train the model and finally profile the traning process for the performance analysis.

 Below we will understand each part of the code in details.
 ### Device Selection:
 This line of code `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` is used to select GPU based on the availability.

 ### Model Initilization:
 This line of code `model = SimpleModel().to(device)` instantiates the `SimpleModel` and moves it to the selected device.

 ### Loss Function:
 `criterion = nn.CrossEntropyLoss()`

We use `CrossEntropyLoss` which is a loss function commanly used for training classification models. For instance in our case (MNIST digit classification). It combines Softamx activation to convert the logits to probabilities and Negative log likelihood(NLL) loss to penalize the incorrect predictions. As discussed earlier, in the model.py section, it is optimized for numerical stability and expects raw logits as inputs. To learn more about the `CrossEntropyLoss` please refer to this site.[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) 

### Optimizer:
`optimizer = optim.Adam(model.parameters())`

We use the Adam optimizer(Adaptive Moment Estimation) which is the most popular opitmization algorithms used in deep learning. It has both Momentum which accelerates convergence by using moving average of gradients and RMSprop which adapts learning rates per-parameter based on squared gradients.One of the other parameter beside the required one`model.parameter()` is learning rate `lr` . If we dont use any value for learning rate it is set to 0.001 by default.

### DataLoader:
 `train_loader = get_mnist_dataloader(batch_size, num_workers)`

 This line of code loads the MNIST training dataset using the `get_mnist_dataloader` function that we defined in the `dataloader.py`


### Profiler Setup:


````python
prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
)
````

We import profile method from `torch.profiler`which profiles the training process to analyze the performance.We can set the if we want to profile the CPU or GPU activities or both. Also we can set the boolean `record_shapes=True` to record the shapes of tensors to help identify the bottlenecks. And `with_stack=True` captures the stack trace for better debugging.

### Training Loop:

````python
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
````

In this training loop, each loop iterates throught the specified number of epochs which in our case is 5. Then before we acutally start iterating over the baches of data, we set the model to the training mode enabling behaviours like Dropout and BatchNorm that are specific to training.

Then the batch loops iterates through the batches of data from the DataLoader.At first the input data and the corresponding labels(target) asre moved to the device where the model is located as they must be on the same device.After this, `  optimizer.zero_grad()` is called to ensure that the gradients are reset to zero before computing the new gradients for the current batch. 

And the purpose of the forward pass here `output = model(data)` is to pass the input data through the model to compute the predictions `output`. After this, loss(error) is computed between the model´s prediction(output) and the true labels(target).Then, there is the backward pass `loss.backward()` to compute the gradients for all model parameters with respect to the loss using backpropagation. 
We also log the current epoch, batch index and loss value every 100 batches for monitoring the training progress using this codde given below
````python
 if batch_idx % 100 == 0:
    print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
````


### Profiler Activation:

We start the profiler at the beginning of epoch 2 to collect performance data for that epoch only using the code as shown  below:

````python
if epoch == 2:  # Profile only epoch 2
    print("Starting profiler...")
    prof.start()
````


### Profiler Stopping and Exporting:

We stop the profiler after epoch 2 and exports the profiling data as a Chrome trace file so that we can visualize it using the Perfetto tool. The code shown below lets us do that.

````python
if epoch == 2 and prof is not None:
    prof.stop()
    print("Profiler stopped. Exporting trace...")
    prof.export_chrome_trace(f"trace_single_gpu_workers_{num_workers}.json")
````


### Main Block
Finally, we call the train function with the `num_workers=4` as default to specify the number of subprocesses for data loading as shown below.Note that, we have passed different value for the `num_workers`from our slurm job script as an argument.

````python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    train(num_workers=args.num_workers)
````
