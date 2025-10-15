import pandas as pd
import numpy as np

def data_load(path):
    df = pd.read_csv(path)
    x = df.drop(["Outcome"], axis=1).to_numpy()
    y = df["Outcome"].to_numpy() 
    x_scaled = scale_data(x)
    return x_scaled, y

def scale_data(x: np.ndarray):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

def train_test_splitting(x, y, test_size=0.25, random=42):
    np.random.seed(random)
    total = len(x)
    idx = np.random.permutation(total)
    test = int(test_size * total)
    test_idx = idx[:test]
    train_idx = idx[test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size=1):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.weight1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weight2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias2 = np.zeros((1, self.output_size))

    def forward_prop(self, x):
        A = np.dot(x, self.weight1) + self.bias1
        Z = sigmoid(A)
        D = np.dot(Z, self.weight2) + self.bias2
        Yp = sigmoid(D)
        return Yp, Z, A, D

    def backward_prop(self, x, y, lr):
        Yp, Z, A, D = self.forward_prop(x)
        N = x.shape[0]
        Y = y.reshape(-1, 1)  

        D2 = Yp - Y
        dEw2 = np.dot(Z.T, D2) / N
        dEb2 = np.sum(D2, axis=0, keepdims=True) / N

        D1 = np.dot(D2, self.weight2.T) * sigmoid_derivative(A)
        dEw1 = np.dot(x.T, D1) / N
        dEb1 = np.sum(D1, axis=0, keepdims=True) / N

        self.weight2 -= lr * dEw2
        self.bias2 -= lr * dEb2
        self.weight1 -= lr * dEw1
        self.bias1 -= lr * dEb1

        loss = np.mean((Yp - Y) ** 2)
        return loss

    def fit(self, x, y, lr=0.05, epochs=10000):
        losses = []
        for e in range(epochs):
            loss = self.backward_prop(x, y, lr)
            losses.append(loss)
            if e % 2000 == 0:
                print(f"Epoch {e}, loss = {loss:.4f}")
        return losses

    def predict(self, x):
        yp, _, _, _ = self.forward_prop(x)
        return yp.astype(np.int64)

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

if __name__ == "__main__":
    x, y = data_load("D:/OneDrive/DME_2_FirstSem/IML/Datasets/diabetes.csv")
    x_train, x_test, y_train, y_test = train_test_splitting(x, y, test_size=0.25, random=42)

    model = NeuralNetwork(input_size=x_train.shape[1], hidden_size=8, output_size=1)
    model.fit(x_train, y_train, lr=0.1, epochs=10000)
    y_pred = model.predict(x_test)
    acc = NeuralNetwork.accuracy(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    