import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Dataset parameters
image_size = (224, 224)
batch_size = 32

# Dataset Paths
train_path = r'D:\studio\archive (4)\Apple\Train'
val_path   = r'D:\studio\archive (4)\Apple\Val'
test_path  = r'D:\studio\archive (4)\Apple\Test'

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_path,
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Class Names:", class_names)

# Preprocessing + Augmentation (ResNet expects preprocess_input)
data_augmentation = tf.keras.Sequential([
    layers.Lambda(preprocess_input),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Build ResNet50 Model
def build_resnet_classifier(input_shape=(224, 224, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)

    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False  # Freeze initially

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, base_model

model, base_model = build_resnet_classifier(num_classes=num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- Training ---
train_start = time.time()
history = model.fit(
    train_ds.prefetch(buffer_size=tf.data.AUTOTUNE),
    validation_data=val_ds.prefetch(buffer_size=tf.data.AUTOTUNE),
    epochs=10,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)
train_time = time.time() - train_start
print(f"Training Time: {train_time:.2f} seconds")

# --- Optional: Fine-tune ResNet ---
# base_model.trainable = True
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-5),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
# model.fit(train_ds, validation_data=val_ds, epochs=5)

# --- Plot Accuracy & Loss ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.tight_layout()
plt.savefig("training_validation_metrics_resnet.png")
plt.show()

# --- Evaluation ---
test_images, test_labels = [], []
for images, labels in test_ds:
    test_images.append(images)
    test_labels.append(labels)
test_images = tf.concat(test_images, axis=0)
test_labels = tf.concat(test_labels, axis=0)

# Inference
inference_start = time.time()
preds = model.predict(test_images)
inference_time = time.time() - inference_start
print(f"Inference Time: {inference_time:.2f} seconds")

# Metrics
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(test_labels.numpy(), axis=1)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

# Save model
model.save('resnet50_apple_leaf_model.h5')

# Print Results
print("\n--- Final Evaluation ---")
print(f"Test Accuracy       : {test_accuracy:.4f}")
print(f"Test Loss           : {test_loss:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted)   : {recall:.4f}")
print(f"F1 Score (weighted) : {f1:.4f}")
print(f"Training Time       : {train_time:.2f} seconds")
print(f"Inference Time      : {inference_time:.2f} seconds")
