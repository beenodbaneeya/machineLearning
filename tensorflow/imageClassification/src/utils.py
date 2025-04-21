from tensorflow.keras.datasets import cifar10
import numpy as np
from config import Config



def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() #the load_data method return X_train,y_train,X_test,y_test.If we check the data using (X_train.shape) we will see 50kk images with 32*32 images 3 RGB channels.test has 10k
    return (
       (X_train.astype('float32') / 255, y_train), # divide each pixel value by 255 as it ranges from 0 to 255.so to normalize into 0 to 1 range we do this.
       (X_test.astype('float32') / 255, y_test) 
    
   )

    

    




