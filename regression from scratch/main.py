import numpy as np
import torch
from functions import *

# input data
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

# target data
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# convert data type from np.array to torch.tensor
inputs = to_torch(inputs)
targets = to_torch(targets)
# print(type(inputs), type(targets))


# num. input features = 3
# num. classes = 2 --> then the weight matrix is 3x2
#                  --> the bias matrix is 2x1


# initializing the weight and the bias matrix
w = torch.randn(3, 2, requires_grad=True)
b = torch.randn(2, requires_grad=True)
# print(w)
# print(b)


# model --> it is a simple linear model with one layer
def model(input_data):
    return input_data @ w + b


# checking the model
outputs = model(inputs)
# print(outputs)


# computing the loss
# loss = loss_fn(outputs=outputs, labels=targets)
# print(loss)


# training for several epochs --> fit function will return the optimized weights and biases
w, b = fit(epochs=200, model=model, inputs=inputs, targets=targets, weights=w, biases=b, learning_rate=1e-5)


# calculate the new losses
preds = model(inputs)
loss = loss_fn(preds, targets)
print(loss)

# checking the final results
print(preds)
print(targets)



















