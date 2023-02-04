import numpy as np
import torch


def to_torch(data):
    data = torch.from_numpy(data)
    return data


def loss_fn(outputs, labels):
    diff = outputs - labels
    return torch.sum(diff*diff) / diff.numel()


def fit(epochs, model, inputs, targets, weights, biases, learning_rate):
    for epoch in range(epochs):
        # getting the predictions
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * learning_rate
            biases -= biases.grad * learning_rate
            weights.grad.zero_()
            biases.grad.zero_()

    return weights, biases
