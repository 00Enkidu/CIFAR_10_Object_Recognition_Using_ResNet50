# CIFAR-10 Object Recognition Using ResNet50

This is a deep learning project using both a custom neural network and transfer learning for image classification. The highlight of this project is the use of **ResNet50**, a powerful convolutional neural network pre-trained on large-scale image datasets, for transfer learning. We demonstrate the effectiveness of transfer learning by comparing it to a custom model built from scratch, applied to the CIFAR-10 dataset.

---

## 1. Dataset Introduction

This project uses the **CIFAR-10** dataset, a widely used benchmark in computer vision for image classification. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The label classes in the dataset are:

- airplane 
- automobile 
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

- **Dataset URL:** [https://www.kaggle.com/c/cifar-10](https://www.kaggle.com/c/cifar-10)

---

## 2. Image Processing Code

The image processing and prediction pipeline is defined as follows (extracted directly from the project code):

```python
def predict_image(image_path):
    img = Image.open(image_path)
    # Resize the image to the input size of the ResNet50 model (32x32)
    img = img.resize((32, 32))
    img = np.array(img)
    # Add a batch dimension to the image array
    img = np.expand_dims(img, axis=0)
    # Normalize the image data
    img = img / 255.0
    # Make the prediction
    prediction = model.predict(img)
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)
    # Map the predicted class index back to the label
    labels_list = list(labels_dictionary.keys())
    predicted_label = labels_list[predicted_class_index]
    return predicted_label
```

---

## 3. Model Architectures

This project evaluates two main architectures:

### a. Custom Model (Baseline)

The first model is a simple custom neural network, serving as a baseline for comparison. Its performance on CIFAR-10 is limited.

**Model Structure (with detailed comments):**
```python
num_of_classes = 10

model = keras.Sequential([
    # The input layer expects images of shape 32x32 pixels with 3 color channels (RGB)
    keras.Input(shape=(32, 32, 3)),
    
    # Flatten the 3D image tensor into a 1D vector (32*32*3 = 3072 features)
    keras.layers.Flatten(),
    
    # First dense (fully connected) layer with 64 neurons and ReLU activation
    keras.layers.Dense(64, activation='relu'),
    
    # Second dense layer with 64 neurons and ReLU activation
    keras.layers.Dense(64, activation='relu'),
    
    # Output layer with `num_of_classes` neurons (10 for CIFAR-10), softmax activation for probability output
    keras.layers.Dense(num_of_classes, activation='softmax')
])
```

- **Input:** 32x32x3 color images
- **Structure:** Flatten → Dense(64, relu) → Dense(64, relu) → Dense(10, softmax)
- **Output:** Probability distribution over the 10 CIFAR-10 classes

---

### b. Transfer Learning Model (ResNet50)

For higher performance, we apply transfer learning using **ResNet50** pre-trained on ImageNet. The model uses ResNet50 as a convolutional base, followed by custom dense layers for classification.

**Model Structure (with detailed comments):**
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers

convolutional_base = ResNet50(
    weights='imagenet',          # Use weights pre-trained on ImageNet
    include_top=False,           # Exclude the fully-connected output layer
    input_shape=(256, 256, 3)    # Input images resized to 256x256 RGB
)

model = models.Sequential()
# Upsample CIFAR-10 images (32x32) to match ResNet50 input size (256x256)
model.add(layers.UpSampling2D((2, 2)))  # 32x32 → 64x64
model.add(layers.UpSampling2D((2, 2)))  # 64x64 → 128x128
model.add(layers.UpSampling2D((2, 2)))  # 128x128 → 256x256

# Add the pre-trained ResNet50 convolutional base
model.add(convolutional_base)

# Flatten the output of the convolutional base
model.add(layers.Flatten())

# Normalize the activations to improve training stability
model.add(layers.BatchNormalization())

# First dense layer with 128 units and ReLU activation
model.add(layers.Dense(128, activation='relu'))
# Dropout to reduce overfitting
model.add(layers.Dropout(0.5))

# Batch normalization again after dropout
model.add(layers.BatchNormalization())

# Second dense layer with 64 units and ReLU activation
model.add(layers.Dense(64, activation='relu'))
# Another dropout layer for regularization
model.add(layers.Dropout(0.5))

# Batch normalization before the output
model.add(layers.BatchNormalization())

# Output layer for classification: 10 units (CIFAR-10), softmax for class probabilities
model.add(layers.Dense(num_of_classes, activation='softmax'))

# Compile the model with a low learning rate suitable for transfer learning
model.compile(
    optimizer=optimizers.RMSprop(learning_rate=2e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
# Training example:
# history = model.fit(X_train_scale, Y_train, validation_split=0.1, epochs=10)
```

---

## 4. Model Training Results and Analysis

### a. Custom Model Training Results

```
Epoch 1/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - acc: 0.2106 - loss: 2.0837 - val_acc: 0.2855 - val_loss: 1.8938
...
Epoch 10/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.3748 - loss: 1.7265 - val_acc: 0.3550 - val_loss: 1.7572
```

#### Analysis

- **Training Accuracy:** Improved from 21% to ~37% over 10 epochs.
- **Validation Accuracy:** Improved from 28% to ~36%.
- **Loss:** Gradual decrease, but overall accuracy is low.

#### Training Result Visualization (Figure 1)

<img width="1010" height="393" alt="image" src="https://github.com/user-attachments/assets/76188ea3-c476-45d4-80a9-aa1ce00c2ef3" />


---

### b. ResNet50 Transfer Learning Model Training Results

```
Epoch 1/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 445s 348ms/step - acc: 0.3149 - loss: 2.1003 - val_acc: 0.7548 - val_loss: 0.9065
...
Epoch 10/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 442s 347ms/step - acc: 0.9733 - loss: 0.1730 - val_acc: 0.9405 - val_loss: 0.2253

313/313 ━━━━━━━━━━━━━━━━━━━━ 40s 113ms/step - acc: 0.9370 - loss: 0.2273
Test Loss: 0.2323
Test Accuracy: 0.9366
```

#### Analysis

- **Transfer learning with ResNet50 achieves over 93% validation and test accuracy.**
- **Significantly outperforms the baseline custom model, confirming the power of transfer learning in deep learning workflows.**

#### Training Result Visualization (Figure 2)

<img width="1001" height="393" alt="image" src="https://github.com/user-attachments/assets/2079bbcb-9a24-4097-9249-b5b52310705e" />

---

## 5. Conclusion

- The baseline custom model shows limited learning capacity on CIFAR-10, with validation accuracy peaking at around 36%. The training and validation accuracy/loss curves indicate the model struggles to fit the complexity of the dataset (See above Figure 1).
- In contrast, the ResNet50-based transfer learning model achieves rapid and significant improvements, reaching over 93% validation and test accuracy within 10 epochs. Its learning curves show strong, steady improvements in both accuracy and loss (See above Figure 2).
- These results visually and quantitatively demonstrate the power of transfer learning: leveraging pre-trained models like ResNet50 dramatically boosts performance on challenging image classification tasks, especially when compared to simple custom architectures.

---

## References

- [CIFAR-10 on Kaggle](https://www.kaggle.com/c/cifar-10)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

---

> **All model code, logs, and result plots are based on the original notebook and project files.  
> For any questions or suggestions, please open an issue.**
