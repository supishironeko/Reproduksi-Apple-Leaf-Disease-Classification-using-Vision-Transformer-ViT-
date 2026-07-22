import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Dataset parameters
image_size = (224, 224)
batch_size = 32

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/jyko/다운로드/hasnain/archive/Apple/Train',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/jyko/다운로드/hasnain/archive/Apple/Val',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/jyko/다운로드/hasnain/archive/Apple/Test',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Patch Encoder Layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

# MLP block
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Build Vision Transformer model
def build_vit_classifier(image_size=224, patch_size=16, projection_dim=64, transformer_layers=8, num_heads=4):
    num_patches = (image_size // patch_size) ** 2
    inputs = layers.Input(shape=(image_size, image_size, 3))
    augmented = data_augmentation(inputs)

    # Create patches
    patches = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size)(augmented)
    patches = layers.Reshape((num_patches, projection_dim))(patches)

    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer blocks
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    features = layers.Dropout(0.5)(representation)
    logits = layers.Dense(num_classes, activation='softmax')(features)

    return tf.keras.Model(inputs=inputs, outputs=logits)

# Create and compile model
model = build_vit_classifier()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- Training ---
train_start = time.time()
history = model.fit(
    train_ds.prefetch(buffer_size=tf.data.AUTOTUNE),
    validation_data=val_ds.prefetch(buffer_size=tf.data.AUTOTUNE),
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)
train_time = time.time() - train_start
print(f"Training Time: {train_time:.2f} seconds")

# --- Plot Accuracy & Loss ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.savefig("training_validation_metrics.png")
plt.show()

# --- Evaluation ---
# Prepare test data
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

# Evaluation Metrics
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(test_labels.numpy(), axis=1)

precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

# --- Save Model ---
model.save('vit_apple_leaf_model.h5')

# --- Print Metrics ---
print("\n--- Final Evaluation ---")
print(f"Test Accuracy     : {test_accuracy:.4f}")
print(f"Test Loss         : {test_loss:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted)   : {recall:.4f}")
print(f"F1 Score (weighted) : {f1:.4f}")
print(f"Training Time       : {train_time:.2f} seconds")
print(f"Inference Time      : {inference_time:.2f} seconds")
