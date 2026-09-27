import numpy as np
from dataset import load_data

#Load data
(X_train, Y_train), (X_test, Y_test), test_labels = load_data()

#Weights and bias
a = 0.1
np.random.seed(42)
W1 = np.random.randn(128, 784) * np.sqrt(2.0 / 784)
b1 = np.zeros((128, 1))
W2 = np.random.randn(64, 128) * np.sqrt(2.0 / 128)
b2 = np.zeros((64, 1))
W3 = np.random.randn(10, 64) * np.sqrt(2.0 / 64)
b3 = np.zeros((10, 1))

#Activate Function
def relu(Z):
    return np.maximum(0, Z)

def relu_deriv(Z):
    return Z > 0

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

#Network

for epoch in range(20):

    permutation = np.random.permutation(X_train.shape[1])
    inputs_shuffled = X_train[:, permutation]
    labels_shuffled = Y_train[:, permutation]

    iteration = 0
    while iteration < X_train.shape[1]:

      inputs_batch = inputs_shuffled[:, iteration : iteration+128]  #784x128
      labels_batch = labels_shuffled[:, iteration : iteration+128]  #10x128
      m_batch = inputs_batch.shape[1] #128
      # Forward Pass (Nhân ma trận)
      Z1 = np.dot(W1, inputs_batch) + b1
      A1 = relu(Z1)

      Z2 = np.dot(W2, A1) + b2
      A2 = relu(Z2)

      Z3 = np.dot(W3, A2) + b3
      output = softmax(Z3)

      # Cross-Entropy Loss
      epsilon = 1e-15
      output_clipped = np.clip(output, epsilon, 1 - epsilon)
      loss = -np.sum(labels_batch * np.log(output_clipped))

      #Backward Pass
      dZ3 = output - labels_batch
      dZ2 = np.dot(W3.T,dZ3)*relu_deriv(Z2)
      dZ1 = np.dot(W2.T, dZ2)*relu_deriv(Z1)

      W3 -= a*(1/m_batch)*np.dot(dZ3, A2.T)
      b3 -= a*(1/m_batch)*np.sum(dZ3, axis=1, keepdims=True)
      W2 -= a*(1/m_batch)*np.dot(dZ2, A1.T)
      b2 -= a*(1/m_batch)*np.sum(dZ2, axis=1, keepdims=True)
      W1 -= a*(1/m_batch)*np.dot(dZ1, inputs_batch.T)
      b1 -= a*(1/m_batch)*np.sum(dZ1, axis=1, keepdims=True)
      iteration += 128

#Test/Forward Pass test

    Z1_t = np.dot(W1, X_test) + b1
    A1_t = relu(Z1_t)
    Z2_t = np.dot(W2, A1_t) + b2
    A2_t = relu(Z2_t)
    Z3_t = np.dot(W3, A2_t) + b3
    A3_t = softmax(Z3_t)

    predictions = np.argmax(A3_t, axis=0)
    accuracy = np.mean(predictions == test_labels) * 100
    print(f"Epoch {epoch+1:2d}/{100} | Độ chính xác tập test: {accuracy:.2f}%")

# 4. Save weights and biaas
np.savez("model_mnist_full.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print("\n--> Hoàn tất! Đã lưu mô hình vào file 'model_mnist_full.npz'")


