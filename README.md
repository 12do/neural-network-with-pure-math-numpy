
# Handwritten Digit Recognition from Scratch (NumPy)

A lightweight, framework-free implementation of a Multi-Layer Perceptron (MLP) built completely from scratch using Python and **NumPy**. This project demonstrates the core mathematical mechanics of Deep Learning—including Forward Propagation, He Initialization, Backpropagation, and Mini-batch Gradient Descent—without relying on high-level libraries like TensorFlow or PyTorch.

The model achieves **~97% test accuracy** on the raw MNIST dataset in 20 epochs.

---

## Key Features

* **Zero Deep Learning Frameworks:** Built purely on NumPy array operations and calculus.
* **Direct Binary I/O:** Custom binary parser for raw MNIST `.ubyte` files.
* **Mathematical Optimization:**
  * **He (Kaiming) Initialization:** Prevents vanishing/exploding gradients in ReLU layers.
  * **Numerically Stable Softmax:** Mitigates floating-point overflow issues.
  * **Mini-Batch Gradient Descent:** Accelerates convergence via vectorized batch updates.
* **Model Serialization:** Exports trained parameters (weights and biases) to a compressed `.npz` file for fast inference.

---

## 📐 Architecture & Mathematical Formulation

The network consists of 4 layers ($784 \rightarrow 128 \rightarrow 64 \rightarrow 10$):

* **Input Layer ($L_0$):** $784$ features (flattened $28 \times 28$ grayscale pixels, normalized to $[0, 1]$).
* **Hidden Layer 1 ($L_1$):** $128$ units with **ReLU** activation.
* **Hidden Layer 2 ($L_2$):** $64$ units with **ReLU** activation.
* **Output Layer ($L_3$):** $10$ units with **Softmax** activation for multi-class classification.

### 1. Forward Propagation

For layer $l \in \{1, 2, 3\}$:

$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$

Where activation functions are:

$$
\text{ReLU}(Z) = \max(0, Z)
$$

$$
\text{Softmax}(Z_i) = \frac{e^{Z_i - \max(Z)}}{\sum_{j} e^{Z_j - \max(Z)}}
$$

### 2. Backpropagation (Gradients Calculation)

Using Cross-Entropy Loss with Softmax output, the error at the final layer is simplified to:

$$
dZ^{[3]} = A^{[3]} - Y
$$

Propagating backward through ReLU hidden layers:

$$
dZ^{[2]} = \left(W^{[3]T} dZ^{[3]}\right) \odot \mathbb{I}(Z^{[2]} > 0)
$$

$$
dZ^{[1]} = \left(W^{[2]T} dZ^{[2]}\right) \odot \mathbb{I}(Z^{[1]} > 0)
$$

Parameter updates for batch size $m$:

$$
W^{[l]} := W^{[l]} - \frac{\alpha}{m} dZ^{[l]} (A^{[l-1]})^T, \quad b^{[l]} := b^{[l]} - \frac{\alpha}{m} \sum dZ^{[l]}
$$


---

## ▶️ Test it by yourself

You can evaluate the trained model on custom handwritten digits (e.g., photos of hand-written numbers or images drawn in Paint/Photoshop) without re-training the network.


### 1. Upload your image

Create a folder named `sample_image` and upload the image you want the model to predict.

### 2. The Inference Script (`test.py`)

Create a file named `test.py` in your project root directory and paste the following implementation:


```python
import numpy as np
from PIL import Image

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Flatten image
image_path = "./sample_image/number9.png"

# 2. Convert into grayscale ('L') and reshape
img = Image.open(image_path).convert('L').resize((28, 28))

# 3. Normalize
img_matrix = np.array(img) / 255.0


X = img_matrix.reshape(img_matrix.shape[0]*img_matrix.shape[1]).astype('float32')
X = img_matrix.reshape(784, 1).astype('float32')
Y = np.zeros((10, 1))         # Shape: (10, 1)
Y[9] = 1.0
a = 0.1

# Trained weights and bias
saved_model = np.load("model_so9.npz")

W1 = saved_model['W1']
b1 = saved_model['b1']
W2 = saved_model['W2']
b2 = saved_model['b2']
W3 = saved_model['W3']
b3 = saved_model['b3']

# Forward Pass 
Z1 = np.dot(W1, X) + b1
A1 = sigmoid(Z1)

Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)

Z3 = np.dot(W3, A2) + b3
exp_z3 = np.exp(Z3 - np.max(Z3))
output = exp_z3 / np.sum(exp_z3)

prediction = np.argmax(output, axis=0)[0]

print(f"Predicted number: {prediction}")
print(f"Confidence: {output[prediction][0] * 100:.2f}%")
```

---

## 📂 Repository Structure

```text
.
├── mnist_dataset/          # Raw MNIST binary files (.ubyte)
├── dataset.py              # Custom binary file parser & preprocessor
├── network.py              # Training script & model implementation
├── test.py                 # Inference script using trained weights (.npz)
├── .gitignore              # Ignores cache and temporary files
├── requirements.txt        # Minimal dependency list (NumPy)
└── README.md               # Project documentation
```
