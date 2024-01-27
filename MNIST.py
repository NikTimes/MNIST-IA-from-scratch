from Layer_Dense import Layer_dense
from Network import NeuralNetwork
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

## Hyperparameters --------------------------------------------------------

input_size = 784  # 28*28
hidden_layer_size = 64
output_size = 10  # MNIST has 10 classes

epochs = 5
batch_size = 32
learning_rate = 0.001

## Creating the neural network --------------------------------------------

# Initialize the network
network = NeuralNetwork()

# Add layers
network.add_layer(Layer_dense(input_size, hidden_layer_size, 'ReLU'))
network.add_layer(Layer_dense(hidden_layer_size, output_size, 'Sigmoid'))

## Training network--------------------------------------------------------
network.train(train_X, train_y, epochs, batch_size, learning_rate)

#Evaluate Network----------------------------------------------------------
test_accuracy = network.evaluate(test_X[1:5000], test_y[1:5000])  #Evaluating subset 5000 element subset of test data 
print(f"Test Accuracy: {test_accuracy*100}%") 

# Select a batch of images and labels for demonstration--------------------
batch_indices = np.random.choice(np.arange(len(test_X)), 25, replace=False)
batch_images = test_X[batch_indices]
batch_true_labels = test_y[batch_indices]

predicted_labels = network.predict(batch_images)
# Reshape for plotting
batch_images = batch_images.reshape(-1, 28, 28)

# Set up the matplotlib----------------------------------------------------
plt.figure(figsize=(10,10))

# Plot the images along with their true and predicted labels
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(batch_images[i], cmap=plt.cm.inferno)
    plt.xlabel(f"True: {batch_true_labels[i]}\nPred: {predicted_labels[i]}")
plt.tight_layout()
plt.show()