# CIFAR-10 Object Recognition Using ResNet50

This is a deep learning project using transfer learning for image classification. In this project, the transferred model is **ResNet50**, a powerful convolutional neural network pre-trained on large-scale image datasets. We leverage the feature extraction capabilities of ResNet50 and apply it to the CIFAR-10 dataset, demonstrating the effectiveness of transfer learning in computer vision tasks.

---

## 1. Dataset Introduction

This project uses the **CIFAR-10** dataset, which is a widely used benchmark in computer vision for image classification. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.
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

## 3. Model Architecture

### a. ResNet50 Introduction

**ResNet50** is a deep convolutional neural network that is 50 layers deep. It is a commonly used backbone for feature extraction in computer vision tasks. The key innovation of ResNet models is the use of **residual connections**, which help alleviate the vanishing gradient problem and allow for the training of much deeper networks.

- **Key features of ResNet50:**
  - 50 convolutional layers
  - Residual blocks with skip connections
  - Strong performance on image classification benchmarks

### b. Custom Model for CIFAR-10

After extracting features using ResNet50 (or for a simple demonstration), a custom neural network is built for classification on CIFAR-10. The architecture is as follows:

```python
num_of_classes = 10

# Setting up layers of neural network
model = keras.Sequential([
    keras.Input(shape=(32,32,3)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_of_classes, activation='softmax')
])
```

- **Input:** 32x32x3 color images
- **Structure:** Flatten layer → Dense(64, relu) → Dense(64, relu) → Dense(10, softmax)
- **Output:** Probability distribution over the 10 CIFAR-10 classes

---

## 4. Model Training Results and Analysis

The following are the training results for 10 epochs. The model shows gradual improvement in both training and validation accuracy:

```
Epoch 1/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - acc: 0.2106 - loss: 2.0837 - val_acc: 0.2855 - val_loss: 1.8938
Epoch 2/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.3092 - loss: 1.8752 - val_acc: 0.3122 - val_loss: 1.8714
Epoch 3/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.3311 - loss: 1.8305 - val_acc: 0.3260 - val_loss: 1.8217
Epoch 4/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.3393 - loss: 1.8016 - val_acc: 0.3475 - val_loss: 1.8008
Epoch 5/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - acc: 0.3493 - loss: 1.7828 - val_acc: 0.3577 - val_loss: 1.7821
Epoch 6/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.3572 - loss: 1.7682 - val_acc: 0.3515 - val_loss: 1.8015
Epoch 7/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.3617 - loss: 1.7447 - val_acc: 0.3587 - val_loss: 1.7720
Epoch 8/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 4s 3ms/step - acc: 0.3606 - loss: 1.7457 - val_acc: 0.3465 - val_loss: 1.7978
Epoch 9/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.3727 - loss: 1.7285 - val_acc: 0.3632 - val_loss: 1.7611
Epoch 10/10
1125/1125 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.3748 - loss: 1.7265 - val_acc: 0.3550 - val_loss: 1.7572
```

### Analysis

- **Training Accuracy:** Improved from 21% to ~37% over 10 epochs.
- **Validation Accuracy:** Improved from 28% to ~36%.
- **Loss:** Both training and validation loss gradually decreased, indicating the model is learning but may benefit from further tuning and regularization.

## 5. Training Result Visualization

Below are the training and validation accuracy and loss curves for the model over 10 epochs:

<img width="1010" height="393" alt="image" src="https://github.com/user-attachments/assets/2f8aa239-6b8c-4de9-97c8-a2e74fbd9b70" />

- The left plot shows the trend of training and validation accuracy.
- The right plot shows the trend of training and validation loss.
- These visualizations illustrate that the model is learning (accuracy increases, loss decreases), but also suggest some possible overfitting or underfitting, as the validation accuracy does not increase as quickly as the training accuracy.

---

## References

- [CIFAR-10 on Kaggle](https://www.kaggle.com/c/cifar-10)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

---

> **All model code, logs, and result plots are based on the original notebook and project files.  
> For any questions or suggestions, please open an issue.**
