# Apple Leaf Disease Classification using Vision Transformer (ViT)

This repository contains a Python script for classifying apple leaf diseases using a Vision Transformer (ViT) model. The dataset used is the Plant Village dataset, which contains images of apple leaves with four classes: Healthy, Apple Scab, Black Rot, and Cedar Apple Rust. The script includes data preprocessing, model training, and evaluation steps.

## Table of Contents
- [Introduction](#introduction)
- [Code Explanation](#code-explanation)
- [Steps for Implementation](#steps-for-implementation)
- [Example Usage](#example-usage)
- [Conclusion](#conclusion)

---

## Introduction

The goal of this project is to classify apple leaf diseases using a Vision Transformer (ViT) model. The dataset is divided into four classes: Healthy, Apple Scab, Black Rot, and Cedar Apple Rust. The script includes data preprocessing, model training, and evaluation steps.

---

## Code Explanation

### 1. **Importing Libraries**
   - The script starts by importing necessary libraries such as `matplotlib`, `seaborn`, `numpy`, `pandas`, `tensorflow`, and `sklearn`. These libraries are used for data visualization, data manipulation, and building/training the deep learning model.

### 2. **Visualizing the Dataset**
   - The `walk_through_dir` function is used to explore the dataset directory structure and count the number of images in each class.
   - The dataset is divided into `Train`, `Val`, and `Test` directories, each containing subdirectories for the four classes.

### 3. **Data Augmentation**
   - The script uses `ImageDataGenerator` from Keras to apply data augmentation techniques such as rotation, horizontal flipping, and rescaling to the training data. This helps in improving the model's generalization ability.
   - Separate generators are created for training, validation, and test datasets.

### 4. **Patch Visualization**
   - The script defines a `Patches` layer that extracts patches from the images. This is a crucial step in Vision Transformers, where images are divided into smaller patches that are then processed by the transformer.
   - The script visualizes these patches for different patch sizes (32x32, 16x16, 8x8) to understand how the image is divided.

### 5. **Model Training**
   - The script defines a Vision Transformer (ViT) model using TensorFlow and Keras. The model is compiled with the Adam optimizer and categorical cross-entropy loss.
   - The model is trained for a specified number of epochs, and the training history is stored for later analysis.

### 6. **Model Evaluation**
   - After training, the model is evaluated on the test dataset. The script generates a confusion matrix and a classification report to assess the model's performance.
   - The confusion matrix is visualized using `seaborn` to provide a clear understanding of the model's predictions.

### 7. **Visualizing Misclassified Images**
   - The script includes functionality to visualize misclassified images, which helps in understanding where the model is making errors.

### 8. **Fine-Tuning and Learning Rate Adjustment**
   - The script demonstrates how to fine-tune the model by adjusting the learning rate and re-training the model.

---

## Steps for Implementation

1. **Dataset Preparation**
   - Ensure that the dataset is organized into `Train`, `Val`, and `Test` directories, with each directory containing subdirectories for each class (Healthy, Apple Scab, Black Rot, Cedar Apple Rust).

2. **Install Required Libraries**
   - Install the necessary Python libraries using pip:
     ```bash
     pip install tensorflow matplotlib seaborn numpy pandas scikit-learn
     ```

3. **Run the Script**
   - Execute the script in a Python environment. The script will automatically:
     - Load and preprocess the dataset.
     - Apply data augmentation.
     - Train the Vision Transformer model.
     - Evaluate the model and generate performance metrics.

4. **Analyze Results**
   - Review the confusion matrix and classification report to understand the model's performance.
   - Visualize misclassified images to identify potential areas for improvement.

5. **Fine-Tuning**
   - Experiment with different patch sizes, learning rates, and data augmentation techniques to improve the model's accuracy.

---

## Example Usage

```python
# Train the model with a specific patch size
model = create_vit_model(patch_size=32)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Evaluate the model on the test set
Y_pred = model.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)
print(classification_report(test_set.classes, y_pred, target_names=class_names))
