# Knowledge Distillation Implementation

This project demonstrates the implementation of Knowledge Distillation using PyTorch on the MNIST dataset. Knowledge Distillation is a model compression technique where a smaller model (student) learns from a larger, more complex model (teacher).

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Performance Analysis](#performance-analysis)
- [Usage](#usage)

## Overview

Knowledge Distillation is a model compression technique introduced by Hinton et al. in 2015. The main idea is to transfer the knowledge from a large, complex model (teacher) to a smaller, simpler model (student). This process allows the student model to achieve comparable performance while being more efficient in terms of computational resources and inference time.

## Requirements

```python
torch
torchvision
numpy
pandas
tqdm
```

## Project Structure

The project consists of two main neural network architectures:

### Teacher Model (TeacherNet)
- Convolutional Neural Network (CNN)
- Architecture:
  - Input Layer (1 channel, 28x28 pixels)
  - Convolutional Layer (32 filters, 5x5 kernel)
  - Max Pooling Layer (5x5)
  - Fully Connected Layer (128 neurons)
  - Output Layer (10 neurons)

### Student Model (StudentNet)
- Simple Feed-forward Neural Network
- Architecture:
  - Input Layer (784 neurons)
  - Hidden Layer (128 neurons)
  - Output Layer (10 neurons)

## Implementation Details

### Data Preprocessing
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

### Knowledge Distillation Loss
The project uses KL (Kullback-Leibler) divergence as the distillation loss:
```python
def knowledge_distillation_loss(student_logits, teacher_logits):
    p_teacher = F.softmax(teacher_logits, dim=1)
    p_student = F.log_softmax(student_logits, dim=1)
    loss = F.kl_div(p_student, p_teacher, reduction='batchmean')
    return loss
```

### Training Process
1. Train the teacher model using standard cross-entropy loss
2. Train the student model using knowledge distillation loss
3. Evaluate both models on the test set

## Results

### Teacher Model Performance
- Final Accuracy: 98.53%
- Training Progress:
  - Epoch 1: 97.60%
  - Epoch 2: 98.00%
  - Epoch 3: 98.44%
  - Epoch 4: 98.24%
  - Epoch 5: 98.53%

### Student Model Performance
- Final Accuracy: 96.34%
- Training Progress:
  - Epoch 1: 93.53%
  - Epoch 2: 94.67%
  - Epoch 3: 96.30%
  - Epoch 4: 96.29%
  - Epoch 5: 96.34%

## Performance Analysis

### Model Size Comparison
- Teacher Model: Complex CNN architecture
- Student Model: Simple feed-forward network
- Size Reduction: Significant reduction in parameters and computational complexity

### Inference Time
- Teacher Model: ~1.61 seconds per evaluation
- Student Model: ~1.09 seconds per evaluation
- Speed Improvement: ~32% faster inference time

### Accuracy vs. Efficiency Trade-off
- Accuracy Drop: ~2.2% (from 98.53% to 96.34%)
- Benefits:
  - Reduced model size
  - Faster inference time
  - Lower computational requirements
  - More suitable for deployment in resource-constrained environments

## Usage

1. Install the required dependencies:
```bash
pip install torch torchvision numpy pandas tqdm
```

2. Run the notebook:
```bash
jupyter notebook Knowledge_Distillation.ipynb
```

3. The notebook will:
   - Load and preprocess the MNIST dataset
   - Train the teacher model
   - Train the student model using knowledge distillation
   - Evaluate and compare both models

## Conclusion

This implementation successfully demonstrates the effectiveness of Knowledge Distillation in model compression. The student model, despite being significantly simpler, achieves comparable performance while offering better computational efficiency. This makes it more suitable for deployment in production environments where resource constraints are a concern.

The project shows that Knowledge Distillation is a powerful technique for creating efficient, deployable models without sacrificing too much in terms of accuracy.
