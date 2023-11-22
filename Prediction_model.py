#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import numpy as np


class RegularizedLinearRegression:
    def __init__(self, learning_rate=0.001, iterations=300000, lambda_param=0.01, num_layers=5, layer_size=64):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.weights = [np.zeros((4, layer_size))] + [np.zeros((layer_size, layer_size)) for _ in range(num_layers-1)]
        #self.weights = None
        self.biases = [0 for _ in range(num_layers)]
        
    def fit(self, X, y):
        # Initialize weights based on the size of the input data
        #self.initialize_weights(X)
        for _ in range(self.iterations):
            self.update_weights(X, y)

    def update_weights(self, X, y):
        num_samples = X.shape[0]

        # Forward pass
        activations = [X]
        for i in range(self.num_layers):
            linear_output = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activations.append(linear_output)

        predicted = activations[-1]

        # Backward pass
        errors = predicted - y
        for i in range(self.num_layers - 1, -1, -1):
            gradient = (np.dot(activations[i].T, errors)) * (1 / num_samples)
            dw = (1 / num_samples) * (np.mean(gradient, axis=0) + self.lambda_param * self.weights[i])
            db = (1 / num_samples) * np.sum(errors)

            # Update weights and biases
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

            # Propagate errors to the previous layer
            errors = np.dot(errors, self.weights[i].T)

    def predict(self, X):
        print("Predict")
        activations = [X]
        for i in range(self.num_layers):
            linear_output = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activations.append(linear_output)
        return activations[-1]
    
    def save_model(self, file_path="model_parameters.npz"):
        # Convert the list of arrays to a NumPy array
        weights_array = np.array(self.weights)
        biases_array = np.array(self.biases)

        # Save the NumPy array along with other parameters
        np.savez(file_path, learning_rate=self.learning_rate, iterations=self.iterations,
             lambda_param=self.lambda_param, num_layers=self.num_layers, layer_size=self.layer_size,
             weights=weights_array, biases=biases_array)


    def load_model(self, file_path="model_parameters.npz"):
        loaded_data = np.load(file_path, allow_pickle=True)
        self.learning_rate = loaded_data['learning_rate']
        self.iterations = loaded_data['iterations']
        self.lambda_param = loaded_data['lambda_param']
        self.num_layers = loaded_data['num_layers']
        self.layer_size = loaded_data['layer_size']
        self.weights = loaded_data['weights'].tolist()
        self.biases = loaded_data['biases'].tolist()
        
        
    ''' def save_model(self, file_path="model_parameters.npz"):
            np.savez(file_path, learning_rate=self.learning_rate, iterations=self.iterations,
                     lambda_param=self.lambda_param, num_layers=self.num_layers, layer_size=self.layer_size,
                     weights=self.weights, biases=self.biases)  '''
            



def main(X_train, Y_train, X_test, Y_test):
    # Instantiate the model
    model = RegularizedLinearRegression()
    
    print("Model Training")
    # Train the model
    model.fit(X_train, Y_train)

    # Make predictions
    print("Make Predictions")
    predictions = model.predict(X_test)


    #Save the model
    model.save_model("trained_model.npz")
    
    error = np.mean(Y_test - predictions)

    print(error)
    

# %%
if __name__ == "__main__":
    data_2021 = pd.read_csv("Data_2021.csv")
    print("Data read")

    print("Separating features and targets")
    Data_features = data_2021.drop(labels="Receipt_Count", axis = 1)
    Data_target = data_2021["Receipt_Count"]
    Data_target = pd.DataFrame(Data_target)
    
    #Train-test split
    print("Train test split")
    X_train = Data_features[:304]
    Y_train = Data_target[:304]
    X_test = Data_features[304:]
    Y_test = Data_target[304:]
    
    print("X_train.shape[1]: ", X_train.shape[1])
    
    #Converting the train and test sets to Numpy arrays.
    print("Converting to numpy arrays")
    train_X = X_train.to_numpy()
    train_Y = Y_train.to_numpy()
    test_X = X_test.to_numpy()
    test_Y = Y_test.to_numpy()
    
    main(train_X, train_Y, test_X, test_Y)
