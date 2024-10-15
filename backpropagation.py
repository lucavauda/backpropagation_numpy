import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        
    def forward(self, X):
        self.z1 = np.dot(X, self.w1)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * sigmoid_derivative(output)
        
        self.z1_error = np.dot(self.output_delta, self.w2.T)
        self.z1_delta = self.z1_error * sigmoid_derivative(self.a1)
        
        self.w1 += np.dot(X.T, self.z1_delta)
        self.w2 += np.dot(self.a1.T, self.output_delta)
    
    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
    def predict(self, X):
        return self.forward(X)

# Example usage
if __name__ == "__main__":
    # Input data
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create and train the network
    nn = SimpleNeuralNetwork(3, 4, 1)
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    # Test the network
    test_data = np.array([[1, 1, 0]])
    prediction = nn.predict(test_data)
    print(f"Prediction for [1, 1, 0]: {prediction[0][0]}")

    # Print final weights
    print("\nFinal weights:")
    print("Input to hidden layer (W1):")
    print(nn.w1)
    print("\nHidden to output layer (W2):")
    print(nn.w2)
