# Image Classification Model Documentation

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
```

## Training Process

### Data Preparation:

1. **Load CIFAR-10 Dataset**:
   ```
   (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
   ```

2. **Normalize Images**:
   ```
   train_images, test_images = train_images / 255.0, test_images / 255.0
   ```

3. **Split Training Data into Training and Validation Sets**:
   ```
   from sklearn.model_selection import train_test_split
   train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
   ```
   
### Data Augmentation:
- Using `ImageDataGenerator` for augmenting the training data to improve generalization.
```
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
```

### Model Compilation:
- Compile the model with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric.
```
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Model Training:
- Train the model with the augmented data for 30 epochs.
```
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    steps_per_epoch=len(train_images) // 32,
    epochs=30,
    validation_data=(val_images, val_labels)
)
```

### Model Evaluation:
- Evaluate the model on the test dataset to get the test accuracy.
```
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
```

## Usage Instructions

### Plot Training and Validation Accuracy/Loss:
Plotting the training and validation accuracy and loss over epochs.
```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

### Save and Load the Model:
Save the trained model to a file.
```
model.save('cnn_image_classification_model.h5')
```

Load the saved model from the file.
```
model = tf.keras.models.load_model('cnn_image_classification_model.h5')
```

### Predict New Images:
Functions to preprocess and predict new images.
```
def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]
```

### Example Prediction:
Predict the class of a new image (uncomment the lines below to test).
```
# image_path = 'path_to_new_image.jpg'
# predicted_class = predict_image(image_path)
# print(f'Predicted class: {predicted_class}')
```

Example prediction on given images:
```
image_path = '/content/Vidwud_faceswap.jpg'
predicted_class = predict_image(image_path)
print(f'Predicted class: {predicted_class}')

image_path = '/content/gosau-3724039.jpg'
predicted_class = predict_image(image_path)
print(f'Predicted class: {predicted_class}')
```
