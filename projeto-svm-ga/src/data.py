from sklearn.datasets import load_digits

def load_dataset():
    data = load_digits()
    X = data.data
    y = data.target
    return X, y
