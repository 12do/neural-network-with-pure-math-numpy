import numpy as np


def load_mnist_images(filename):
  with open(filename, 'rb') as f:
    _ = f.read(16) 
    data = np.fromfile(f, dtype=np.uint8)
  return data.reshape(-1, 784)


def load_mnist_labels(filename):
  with open(filename, 'rb') as f:
    _ = f.read(8) 
    data = np.fromfile(f, dtype=np.uint8)
  return data


def load_data(
    train_img_path='./mnist_dataset/train-images-idx3-ubyte',
    train_lbl_path='./mnist_dataset/train-labels-idx1-ubyte',
    test_img_path='./mnist_dataset/t10k-images-idx3-ubyte',
    test_lbl_path='./mnist_dataset/t10k-labels-idx1-ubyte',
):
    # Read file
    X_train_raw = load_mnist_images(train_img_path)
    train_labels = load_mnist_labels(train_lbl_path)
    X_test_raw = load_mnist_images(test_img_path)
    test_labels = load_mnist_labels(test_lbl_path)

    # Reshape & Standardize (784 x N)
    X_train = X_train_raw.T / 255.0 #784x60000
    X_test = X_test_raw.T / 255.0   #784x10000

    # One-hot encoding (10 x N)
    Y_train = np.eye(10)[train_labels].T    #10x60000
    Y_test = np.eye(10)[test_labels].T      #10x10000

    return (X_train, Y_train), (X_test, Y_test), test_labels