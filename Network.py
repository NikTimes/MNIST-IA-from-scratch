import numpy as np 
from tqdm import tqdm
from Utils import batch, one_hot_encode

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.Forward(inputs)
        return inputs
    
    def backward(self, loss_gradient):
        for layer in reversed(self.layers):
            loss_gradient = layer.BackProp(loss_gradient)
    
    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.Forward(inputs)
        return np.argmax(inputs, axis=1)

    def calculate_accuracy(self, predictions, labels):
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(labels, axis=1)
        correct = np.sum(pred_labels == true_labels)
        acc = correct / len(labels)
        return acc
    
    def evaluate(self, test_X, test_Y):
        output = self.forward(test_X)
        predictions = np.argmax(output, axis=1)
        num_classes = output.shape[0]  
        
        predictions_encoded = one_hot_encode(predictions, num_classes)
        test_Y_encoded = one_hot_encode(test_Y, num_classes)  

        evaluation = self.calculate_accuracy(predictions_encoded, test_Y_encoded)
        return evaluation
    
    def train(self, train_X, train_Y, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            total_batches = 0

            progress_bar = tqdm(batch(train_X, train_Y, batch_size), total=len(train_X) // batch_size)

            for batch_x, batch_y in progress_bar:
                batch_x = batch_x.reshape(batch_x.shape[0], -1) / 255.0  # Flatten and normalize
        
                output = self.forward(batch_x)
                batch_y_encoded = one_hot_encode(batch_y, len(output[0]))

                loss = -np.sum(batch_y_encoded * np.log(output + 1e-7)) / batch_size
                epoch_loss += loss

                batch_accuracy = self.calculate_accuracy(output, batch_y_encoded)
                epoch_accuracy += batch_accuracy

                error_value = output - batch_y_encoded
                self.backward(error_value)
                self.update_parameters(learning_rate)

                total_batches += 1
                progress_bar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss:.4f} Accuracy: {batch_accuracy:.4f}")
            
        epoch_loss /= total_batches
        epoch_accuracy /= total_batches

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss:.4f}, Average Accuracy: {epoch_accuracy:.4f}")
