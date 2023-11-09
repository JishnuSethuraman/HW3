import numpy as np

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1, self.b1, self.W2, self.b2 = self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize weights and biases
        W1 = np.random.randn(self.hidden_dim, self.input_dim) * 0.01
        b1 = np.zeros((self.hidden_dim, 1))
        W2 = np.random.randn(self.output_dim, self.hidden_dim) * 0.01
        b2 = np.zeros((self.output_dim, 1))
        return W1, b1, W2, b2

    def sigmoid(self, Z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-Z))

    def forward_propagation(self, X):
        # Forward propagation
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)
        return Z1, A1, Z2, A2

    def compute_cost(self, A2, Y):
        # Compute the cost
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A2) + (1-Y) * np.log(1-A2))
        cost = np.squeeze(cost)  # makes sure cost is the dimension we expect. 
        return cost

    def backpropagation(self, X, Y, Z1, A1, Z2, A2):
        grads = {}
        m = X.shape[1]

        # Compute gradients
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        return grads

    def update_parameters(self, grads, learning_rate):
        # Update weights and biases
        self.W1 -= learning_rate * grads['dW1']
        self.b1 -= learning_rate * grads['db1']
        self.W2 -= learning_rate * grads['dW2']
        self.b2 -= learning_rate * grads['db2']

    def train(self, X, Y, learning_rate, num_iterations):
        for i in range(num_iterations):
            Z1, A1, Z2, A2 = self.forward_propagation(X)
            cost = self.compute_cost(A2, Y)
            grads = self.backpropagation(X, Y, Z1, A1, Z2, A2)
            self.update_parameters(grads, learning_rate)
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")


if __name__ == "__main__":
    # Generate synthetic data
    X = np.random.rand(2, 500)
    Y = np.random.randint(0, 2, size=(1, 500))

    # Initialize the neural network
    nn = SimpleNN(2, 4, 1)

    # Train the neural network
    nn.train(X, Y, learning_rate=0.01, num_iterations=1000)
