# Apple Leaf Disease Classification using Vision Transformer (ViT)

This project implements a Vision Transformer (ViT) model for classifying apple leaf diseases using the Plant Village dataset. The experiment focuses on analyzing the effect of different patch sizes in Vision Transformer architecture to determine the optimal configuration for apple leaf disease classification.

The experiment evaluates three different patch sizes:

- Patch Size 32×32
- Patch Size 16×16
- Patch Size 8×8

The objective of this experiment is to analyze the trade-off between feature representation capability and computational complexity in Vision Transformer models.

---

# Dataset

The dataset used in this experiment is the **Plant Village Apple Leaf Disease Dataset**.

The dataset consists of four classes:
Apple Scab
Black Rot
Cedar Apple Rust
Healthy

The dataset is divided using a hold-out strategy:

| Dataset | Percentage |
|---------|------------|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The dataset structure:
dataset/
<img width="1254" height="1254" alt="struktur folder" src="https://github.com/user-attachments/assets/6b42f6ea-dd57-45b8-b13e-02c8d320f899" />



---

# Environment

The experiment was conducted using:

- Google Colab
- GPU acceleration
- TensorFlow / Keras framework

GPU acceleration was utilized to reduce training time, especially for Vision Transformer models with smaller patch sizes.

---

# Data Preprocessing

Before training, all images are resized into:
224 × 224 pixels


Data augmentation techniques were applied to improve model generalization:

- Random Horizontal Flip
- Random Rotation
- Random Zoom
- Image Rescaling

The validation and testing datasets only use resizing and normalization without augmentation.

---

# Vision Transformer Architecture

The Vision Transformer (ViT) divides an input image into multiple smaller patches. Each patch is treated as a token and processed using Transformer Encoder blocks.

The implemented ViT consists of:

1. Patch Embedding
2. Positional Embedding
3. Multi-Head Self Attention
4. Transformer Encoder Blocks
5. MLP Classification Head

The model configuration:

| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 |
| Projection Dimension | 256 |
| Transformer Layers | 6 |
| Attention Heads | 8 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Activation | Softmax |

---

# Patch Size Experiment

Three different patch sizes were evaluated.

## Patch Size 32×32

A larger patch size produces fewer image tokens:
224 / 32 = 7

Total tokens:
7 × 7 = 49 tokens


Advantages:
- Faster training
- Lower computational cost

Disadvantages:
- Less detailed feature representation
- Small disease patterns may be missed

---

## Patch Size 16×16

Patch size 16 produces:
224 / 16 = 14

Total tokens:
14 × 14 = 196 tokens


This configuration provides a balance between:

- Local feature extraction
- Global feature understanding
- Computational efficiency

---

## Patch Size 8×8

Patch size 8 produces:
224 / 8 = 28

Total tokens:
28 × 28 = 784 tokens


Although smaller patches provide more detailed information, the number of tokens significantly increases the computational complexity of the Transformer self-attention mechanism.

The experiment could not be completed due to limited computational resources on Google Colab GPU.

---

# Experimental Results

## Performance Comparison

| Patch Size | Accuracy | Precision | Recall | F1 Score | Training Time |
|------------|----------|-----------|--------|----------|---------------|
| 32×32 | 90.92% | 91.95% | 90.92% | 90.88% | 300.79 s |
| 16×16 | **97.86%** | **97.87%** | **97.86%** | **97.86%** | 1423.14 s |
| 8×8 | - | - | - | - | Failed |

---

# Result Analysis

## Patch Size 32×32

The model achieved 90.92% accuracy. The lower performance compared to smaller patches occurs because larger patches contain fewer tokens, resulting in reduced local information. Some disease patterns on leaf surfaces may not be captured effectively.

However, this configuration provides the fastest training time due to lower computational requirements.

---

## Patch Size 16×16

Patch size 16×16 achieved the highest performance with:

- Accuracy: 97.86%
- Precision: 97.87%
- Recall: 97.86%
- F1 Score: 97.86%

This result indicates that patch size 16×16 provides the optimal balance between extracting detailed leaf disease patterns and maintaining manageable computational complexity.

---

## Patch Size 8×8

The patch size 8×8 experiment generated 784 tokens per image. The increased number of tokens significantly increased the memory requirement and computation time of the Transformer attention mechanism.

The experiment was unable to complete because of computational limitations.

---

# Conclusion

Based on the experimental results, the Vision Transformer with **patch size 16×16** achieved the best classification performance for apple leaf disease detection.

The comparison shows:

- Patch 32×32:
  - Faster computation
  - Lower accuracy due to limited local feature extraction

- Patch 16×16:
  - Highest accuracy
  - Best balance between performance and computational efficiency

- Patch 8×8:
  - Higher feature detail
  - Requires significantly larger computational resources

Therefore, **ViT Patch Size 16×16 was selected as the optimal configuration for apple leaf disease classification.**

