# 🧠 Deep Learning Models for Image Classification

### CNN + LSTM vs Vision Transformer (ViT)

---

## Overview

This project focuses on implementing and comparing two deep learning architectures for image classification:

* **CNN + LSTM (Hybrid Model)**
* **Vision Transformer (ViT)**

The objective is to analyze how different architectures handle spatial and global feature learning in images.

---

## Objective

* Design and implement deep learning models for image classification
* Compare performance using standard evaluation metrics
* Understand differences between sequential and attention-based models

---

## Dataset

* **Dataset:** Intel Image Classification Dataset

* **Classes:**

  * Buildings
  * Forest
  * Glacier
  * Mountain
  * Sea
  * Street

* **Total Images:**

  * Training: ~14,000
  * Testing: ~3,000

---

## Data Preprocessing

* Resize images to fixed dimensions
* Convert images into tensors
* Apply augmentation (horizontal flip)

### Transformation:

$$
(H, W, C) \rightarrow (C, H, W)
$$

---

## Model 1: CNN + LSTM

### CNN Feature Extraction

* Convolution operation:
  $$
  (I * K)(x,y) = \sum I(x+i, y+j)K(i,j)
  $$

* Architecture:

  * Conv → ReLU → MaxPooling (×3)

* Feature Map:
  $$
  (B, 128, 28, 28)
  $$

---

### Sequence Conversion

```python
x = x.view(B, C, H*W)
x = x.permute(0, 2, 1)
```

* Converts feature map into sequence:
  $$
  (B, 128, 28 \times 28) \rightarrow (B, 784, 128)
  $$

---

### LSTM Processing

* Hidden state update:
  $$
  h_t = \sigma(Wx_t + Uh_{t-1})
  $$

* Captures relationships between spatial regions

---

### Final Output

$$
y = \text{Softmax}(Wx + b)
$$

---

### Limitations

* Loss of spatial structure
* High sequence length (784 steps)
* Less effective global feature modeling

---

## Model 2: Vision Transformer (ViT)

### 🔹 Core Idea

* Image is split into patches:
  $$
  Image \rightarrow Patches \rightarrow Sequence
  $$

---

### Self-Attention Mechanism

$$
Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

* Each patch attends to all others
* Captures global dependencies

---

### Advantages

* Global feature understanding
* No spatial information loss
* Better generalization

---

## Training Details

* Loss Function:
  $$
  L = -\sum y \log(\hat{y})
  $$

* Optimizer:
  $$
  \theta = \theta - \alpha \cdot \nabla L
  $$

* Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

## Results

### CNN + LSTM

* Accuracy: **73.57%**
* Macro F1-score: **0.71**

---

### Vision Transformer (ViT)

* Accuracy: **93.56%**
* Macro F1-score: **0.94**

---

## Model Comparison

| Model      | Accuracy   | F1 Score | Feature Learning   |
| ---------- | ---------- | -------- | ------------------ |
| CNN + LSTM | 73.57%     | 0.71     | Local + Sequential |
| ViT        | **93.56%** | **0.94** | Global             |

---

## Key Insights

* CNN + LSTM:

  * Good for local + sequential patterns
  * Struggles with global dependencies

* ViT:

  * Uses attention to capture full-image context
  * Significantly higher performance

---

## Conclusion

The Vision Transformer significantly outperforms the CNN+LSTM model due to its ability to capture global dependencies using self-attention.

$$
\text{ViT} > \text{CNN+LSTM}
$$

Transformer-based architectures are more effective for complex image classification tasks.

---

## Future Work

* Fine-tuning the full ViT model
* Using hybrid CNN + Transformer models
* Applying advanced data augmentation

---

## Tech Stack

* PyTorch
* Transformers (HuggingFace)
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn

---

##  Author

**Arun Kumar**

---
