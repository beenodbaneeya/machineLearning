from tensorflow.keras import layers, models
from config import Config

def create_cnn_model():
    model = models.Sequential([
        # convulation layer detect the feature. it will find the best filter for us, we just need to define the filter size, and number of filter from us.
        # since we put 32, it will detect 32 different features or edges in our image. and the size is 3*3. then the input_shape is 32*32*3 which is the shape of one image

        layers.Conv2D(32,(3,3), activation='relu', input_shape=Config.INPUT_SHAPE), # when we are in the middle layer,we dont need to specify the shape as network can figure it out automatically
        layers.MaxPooling2D((2,2)),  # popular method, we can use average pooling too
        layers.Conv2D(64,(3,3), activation='relu'), # we will keep one dense network because CNN would have done most of the work.So we dont need so many neurons and many deep layers.
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(Config.NUM_CLASSES, activation='softmax') # we use softmax which is one of the popular activation function.it will normalize our probability. Using sfotmax ensure if you add all the value the sum will be 1 but, sigmoid doesnot do that.

    ])

    model.compile(
        optimizer='adam', #other optimizers are SGD, n-adam .we choose adam
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



# Input layer is the flatten layer which is the first layer which has the input_shape of 32,32,3
# then we have two deep layers one having 3k neurons and other having 1k and the last layer is having 10 categories
# because we have total 10 categories.And the optimizer is SGD and loss is sparse_categorical_crossentropy
# when we have categories and our y is one-hot encoded we can use categorical_crossentropy. But if our y is directly a 
# value. for instance :8 then we have to use spare_categorical_crossentropy




# CNN will have convulational layer, then we have ReLu which is activation and then pooling.
# you could have again the same layers like beforea and at the end you have a dense network

