# CIFAR-10 Image Classification Model Documentation

## Model Architecture

This model is a Convolutional Neural Network (CNN) designed for image classification on the CIFAR-10 dataset. The architecture consists of several convolutional layers followed by dense layers for classification.

### Layers:

1. **Input Layer**:
   - **Input Shape**: (32, 32, 3) for CIFAR-10 images.

2. **Convolutional Layers**:
   - **Conv2D Layer 1**: 32 filters, kernel size (3, 3), ReLU activation.
   - **MaxPooling2D Layer 1**: Pool size (2, 2).
   - **Conv2D Layer 2**: 64 filters, kernel size (3, 3), ReLU activation.
   - **MaxPooling2D Layer 2**: Pool size (2, 2).
   - **Conv2D Layer 3**: 64 filters, kernel size (3, 3), ReLU activation.
   - **MaxPooling2D Layer 3**: Pool size (2, 2).

3. **Dense Layers**:
   - **Flatten Layer**: Converts 3D feature maps to 1D feature vectors.
   - **Dense Layer 1**: 64 units, ReLU activation.
   - **Output Layer**: 10 units (one for each class), Softmax activation.

### Model Summary:

```python
model.summary()
