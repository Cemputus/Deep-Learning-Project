# 🌱 Bean Leaf Disease Detection Using Vision Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cemputus/Deep-Learning-Project/blob/main/image_classification_vit_base_and_advanced.ipynb)

A deep learning solution for automated bean leaf disease detection using Vision Transformers (ViT). This project implements and compares two ViT models to classify bean leaves into three categories: Healthy, Angular Leaf Spot, and Bean Rust.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Sustainable Development Goals (SDGs)](#sustainable-development-goals-sdgs)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Gradio Demo](#gradio-demo)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project addresses the critical challenge of early disease detection in bean crops, which is essential for food security and economic stability, particularly in Uganda. By leveraging state-of-the-art Vision Transformer models, we provide an automated, accessible solution for farmers to identify diseases in bean leaves without requiring expert agricultural knowledge.

### Key Features

- **Two Model Comparison**: Baseline ViT (trained from scratch) vs Advanced ViT (transfer learning)
- **High Accuracy**: Advanced model achieves **95.31% accuracy** on test set
- **Comprehensive Evaluation**: Detailed metrics including precision, recall, F1-score, confusion matrices, and ROC curves
- **Interactive Demo**: Gradio-based web application for real-world use
- **Well-Documented**: Extensive exploratory data analysis and visualization

## 🔍 Problem Statement

Bean crops in Uganda face significant threats from diseases such as Angular Leaf Spot and Bean Rust, which can severely impact crop yields and food security. Manual inspection by agricultural experts is:

- **Time-consuming**: Requires expert knowledge and physical presence
- **Expensive**: Not always accessible to smallholder farmers
- **Limited Availability**: Experts may not be available in remote farming areas
- **Subjective**: Human judgment can vary between inspectors

### Solution Approach

We develop automated disease detection using Vision Transformers (ViT), a state-of-the-art deep learning architecture that treats images as sequences of patches, similar to how transformers process text. This approach enables:

- **Automated Detection**: Instant classification without human intervention
- **Accessibility**: Available to farmers in remote areas via mobile devices
- **Consistency**: Objective, reproducible results
- **Early Detection**: Identify diseases before they spread extensively

## 🎯 Objectives

1. **Data Collection & Analysis**

   - Load and analyze the beans dataset from Hugging Face
   - Perform comprehensive exploratory data analysis (EDA)
   - Visualize class distributions and image characteristics
2. **Model Development**

   - Build a baseline ViT model trained from scratch
   - Build an advanced ViT model using transfer learning
   - Implement proper data preprocessing and augmentation pipelines
3. **Model Evaluation**

   - Compare both models using standard evaluation metrics
   - Generate confusion matrices and ROC curves
   - Perform detailed error analysis
4. **Deployment**

   - Create a Gradio demo application for real-world use
   - Implement open-set detection for unknown/non-bean images
   - Optimize for practical deployment scenarios

## 🌍 Sustainable Development Goals (SDGs)

This project directly contributes to multiple United Nations Sustainable Development Goals:

- **SDG 2 - Zero Hunger**: Increasing crop yields through early disease detection
- **SDG 1 - No Poverty**: Beans are key income crops for smallholder farmers
- **SDG 9 - Industry, Innovation & Infrastructure**: Use of Computer Vision Models (ViT) demonstrates digital agriculture and innovation-driven development
- **SDG 12 - Responsible Consumption**: Reduce unnecessary pesticide use through targeted treatment
- **SDG 13 - Climate Action**: Climate variability increases crop disease prevalence, making early detection crucial

## 📊 Dataset

### Dataset Source

The **Beans** dataset is sourced from Hugging Face, collected by the Makerere AI Lab and NaCRRI (National Crops Resources Research Institute). This dataset contains labeled images of bean leaves with three classes.

**Dataset Link**: [https://huggingface.co/datasets/beans](https://huggingface.co/datasets/beans)

### Dataset Statistics

| Split                | Angular Leaf Spot | Bean Rust | Healthy | Total |
| -------------------- | ----------------- | --------- | ------- | ----- |
| **Train**      | 345               | 348       | 341     | 1,034 |
| **Validation** | 44                | 45        | 44      | 133   |
| **Test**       | 43                | 43        | 42      | 128   |
| **Total**      | 432               | 436       | 427     | 1,295 |

### Dataset Characteristics

- **Image Size**: 500×500 pixels (uniform across all images)
- **Class Balance**: Well-balanced dataset (~33% per class)
- **Format**: RGB images (PIL Image objects)
- **Labels**: Integer labels (0, 1, 2) representing:
  - `0`: Angular Leaf Spot
  - `1`: Bean Rust
  - `2`: Healthy

### Visual Characteristics

- **Angular Leaf Spot**: Irregular brown patches on leaves
- **Bean Rust**: Circular brown spots surrounded by white-ish yellow rings
- **Healthy**: Uniform green color with no visible disease symptoms

## 🔬 Methodology

### Data Preprocessing

1. **Image Processing**

   - Resize images to 224×224 pixels (ViT standard input size)
   - Normalize pixel values using ImageNet statistics
   - Convert to tensor format for model input
2. **Data Augmentation** (Training Set Only)

   - Random horizontal flip (p=0.5)
   - Random vertical flip (p=0.3)
   - Random rotation (±30 degrees)
   - Color jitter (brightness, contrast, saturation, hue)
   - Random affine transformations (translation, scaling)
3. **Data Splits**

   - Training: 1,034 images
   - Validation: 133 images
   - Test: 128 images

### Model Training Strategy

#### Baseline Model

- **Architecture**: Smaller ViT configuration
  - Hidden size: 384 (vs 768 in base model)
  - Number of layers: 6 (vs 12 in base model)
  - Attention heads: 6 (vs 12 in base model)
- **Training**: From scratch (no pre-trained weights)
- **Epochs**: 10
- **Learning Rate**: 5e-4
- **Batch Size**: 16
- **Parameters**: ~11 million

#### Advanced Model

- **Architecture**: Pre-trained ViT-Base
  - Model: `google/vit-base-patch16-224-in21k`
  - Pre-trained on ImageNet-21k (14 million images, 21,843 classes)
- **Training**: Fine-tuning (transfer learning)
- **Epochs**: 4
- **Learning Rate**: 2e-4 (lower for fine-tuning)
- **Batch Size**: 16
- **Parameters**: ~86 million

### Training Configuration

- **Optimizer**: AdamW (default in Hugging Face Trainer)
- **Mixed Precision**: FP16 (when GPU available)
- **Evaluation Strategy**: Every 50 steps
- **Model Checkpointing**: Save best model based on validation accuracy
- **Early Stopping**: Enabled (load best model at end)

## 🏗️ Model Architecture

### Vision Transformer (ViT) Overview

Vision Transformers treat images as sequences of patches:

1. **Patch Embedding**: Split image into fixed-size patches (16×16 pixels)
2. **Linear Projection**: Embed each patch with a linear projection
3. **Position Embedding**: Add learnable position embeddings
4. **Transformer Encoder**: Process through multiple transformer encoder layers
5. **Classification Head**: Use [CLS] token for final classification

### Model Specifications

| Component                   | Baseline Model | Advanced Model         |
| --------------------------- | -------------- | ---------------------- |
| **Base Architecture** | Custom ViT     | ViT-Base (Pre-trained) |
| **Image Size**        | 224×224       | 224×224               |
| **Patch Size**        | 16×16         | 16×16                 |
| **Hidden Size**       | 384            | 768                    |
| **Layers**            | 6              | 12                     |
| **Attention Heads**   | 6              | 12                     |
| **Intermediate Size** | 1536           | 3072                   |
| **Parameters**        | 11M            | 86M                    |
| **Pre-training**      | None           | ImageNet-21k           |

## 📈 Results

### Overall Performance Comparison

| Metric              | Baseline Model | Advanced Model   | Improvement |
| ------------------- | -------------- | ---------------- | ----------- |
| **Accuracy**  | 75.00%         | **95.31%** | +20.31%     |
| **Precision** | 75.31%         | **95.56%** | +20.25%     |
| **Recall**    | 75.00%         | **95.31%** | +20.31%     |
| **F1-Score**  | 74.75%         | **95.30%** | +20.55%     |
| **Test Loss** | 0.6572         | **0.1691** | -74.3%      |

### Per-Class Performance (Advanced Model)

| Class                       | Precision | Recall  | F1-Score |
| --------------------------- | --------- | ------- | -------- |
| **Angular Leaf Spot** | 89.58%    | 100.00% | 94.51%   |
| **Bean Rust**         | 97.44%    | 88.37%  | 92.68%   |
| **Healthy**           | 100.00%   | 97.62%  | 98.80%   |

### Error Analysis

- **Baseline Model Errors**: 28/128 (21.88% error rate)
- **Advanced Model Errors**: 6/128 (4.69% error rate)
- **Most Common Confusions**:
  - Baseline: Bean Rust ↔ Angular Leaf Spot (14 cases)
  - Advanced: Bean Rust → Angular Leaf Spot (5 cases)

### Key Insights

1. **Transfer Learning Advantage**: The advanced model shows a **20+ percentage point improvement** across all metrics, demonstrating the power of pre-trained models.
2. **Training Efficiency**: The advanced model achieved superior results in **4 epochs** compared to the baseline's **10 epochs**, showing faster convergence.
3. **Generalization**: The advanced model exhibits better generalization with fewer misclassifications and more confident predictions.
4. **Class-Specific Performance**: The advanced model achieves near-perfect performance on healthy leaves (100% precision) and excellent performance on disease classes.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/Cemputus/Deep-Learning-Project.git
cd Deep-Learning-Project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install datasets transformers torch torchvision pillow matplotlib seaborn scikit-learn evaluate gradio accelerate
```

### Step 3: Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 💻 Usage

### Running the Notebook

1. **Open in Google Colab** (Recommended):

   - Click the "Open In Colab" badge at the top
   - Enable GPU runtime: Runtime → Change runtime type → GPU
2. **Run Locally**:

   ```bash
   jupyter notebook image_classification_vit_base_and_advanced.ipynb
   ```

### Training Models

The notebook is organized into sections:

1. **Data Loading & EDA**: Load dataset and perform exploratory analysis
2. **Data Preprocessing**: Set up preprocessing pipelines
3. **Model Development**: Create baseline and advanced models
4. **Training**: Train both models (⚠️ Takes 30-60 minutes on GPU)
5. **Evaluation**: Evaluate models and generate metrics
6. **Visualization**: Create plots and confusion matrices
7. **Gradio Demo**: Launch interactive web application

### Quick Start - Inference Only

If you want to use a pre-trained model for inference:

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load model and processor
model = ViTForImageClassification.from_pretrained("path/to/model")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Load and preprocess image
image = Image.open("path/to/bean_leaf.jpg")
inputs = processor(image, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()

class_names = ['angular_leaf_spot', 'bean_rust', 'healthy']
print(f"Prediction: {class_names[pred_id]}")
print(f"Confidence: {probs[0][pred_id]:.2%}")
```

## 📊 Evaluation Metrics

### Metrics Used

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class classification breakdown
- **ROC Curves**: Receiver Operating Characteristic curves
- **Precision-Recall Curves**: PR curves for each class

### Evaluation Results Visualization

The notebook includes:

- Confusion matrices (raw and normalized)
- ROC curves for each class
- Precision-Recall curves
- Training loss/accuracy curves
- Per-class performance bar charts
- Misclassified example visualizations

## 🎨 Gradio Demo

### Launching the Demo

The project includes a Gradio-based web application for interactive disease detection:

```python
# Run the Gradio demo cell in the notebook
demo.launch(share=True)  # Creates a public URL
```

### Demo Features

- **Image Upload**: Drag-and-drop or click to upload bean leaf images
- **Real-time Prediction**: Instant classification results
- **Confidence Scores**: Display prediction confidence
- **Class Probabilities**: Show probability distribution across all classes
- **Open-Set Detection**: Identifies non-bean images as "Unknown" (threshold: 0.68)

### Demo Interface

```
🌱 Bean Leaf Disease Detector

Upload an image of a bean leaf to detect whether it is Healthy,
Angular Leaf Spot, or Bean Rust.
If the image is not a bean leaf, the system labels it as 'Unknown'.
```

## 🔑 Key Findings

### 1. Transfer Learning Superiority

The advanced model (pre-trained ViT) significantly outperforms the baseline model:

- **20+ percentage points** improvement in all metrics
- **4.69% error rate** vs 21.88% for baseline
- Demonstrates the value of pre-trained models for small datasets

### 2. Training Efficiency

- Advanced model converges in **4 epochs** vs baseline's **10 epochs**
- Faster training time despite larger model size
- Lower computational cost per accuracy point

### 3. Robust Feature Learning

- Pre-trained weights provide strong general-purpose visual features
- Fine-tuning adapts these features to bean leaf disease characteristics
- Better handling of edge cases and ambiguous images

### 4. Practical Applicability

- High accuracy (95.31%) suitable for real-world deployment
- Fast inference time (~100 images/second on GPU)
- Mobile-friendly model size (~86M parameters)

## ⚠️ Limitations

1. **Dataset Size**: Relatively small dataset (1,295 images) may limit generalization to diverse real-world conditions
2. **Image Quality Dependency**: Model performance depends on:

   - Image quality and resolution
   - Lighting conditions
   - Camera angles and perspectives
   - Background complexity
3. **Class Distribution**: While balanced in training, real-world scenarios may have different distributions
4. **Limited Disease Coverage**: Only detects three classes; may miss other diseases or conditions
5. **Environmental Factors**: Model trained on specific dataset may not generalize to different:

   - Bean varieties
   - Growing conditions
   - Geographic regions
   - Camera types
6. **Computational Requirements**: Advanced model requires GPU for efficient training (though CPU inference is possible)

## 🔮 Future Work

### Short-term Improvements

1. **Enhanced Data Augmentation**

   - More aggressive augmentation strategies
   - Synthetic data generation (GANs, diffusion models)
   - Domain adaptation techniques
2. **Model Optimization**

   - Knowledge distillation for smaller models
   - Quantization for mobile deployment
   - Pruning for faster inference
3. **Ensemble Methods**

   - Combine multiple models for improved accuracy
   - Voting or weighted averaging strategies
   - Cross-validation ensemble

### Long-term Enhancements

1. **Expanded Dataset**

   - Collect more diverse images
   - Include multiple bean varieties
   - Add temporal progression data (disease stages)
2. **Multi-Disease Detection**

   - Extend to additional diseases
   - Support multiple crop types
   - Severity estimation (mild, moderate, severe)
3. **Real-time Processing**

   - Video processing capabilities
   - Mobile app development
   - Edge device deployment
4. **Advanced Architectures**

   - Experiment with ViT-Large or ViT-Huge
   - Try Swin Transformer or ConvNeXt
   - Explore hybrid CNN-Transformer models
5. **Integration & Deployment**

   - Agricultural advisory system integration
   - SMS/WhatsApp bot for farmers
   - Cloud-based API service
6. **Explainability**

   - Attention visualization
   - Grad-CAM or similar techniques
   - Model interpretability analysis

## 📚 References

1. **Vision Transformer Paper**

   - Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *arXiv preprint arXiv:2010.11929*
2. **Dataset**

   - Beans Dataset: [https://huggingface.co/datasets/beans](https://huggingface.co/datasets/beans)
   - Makerere AI Lab & NaCRRI: Dataset contributors
3. **Libraries & Tools**

   - Hugging Face Transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
   - ViT Model: [https://huggingface.co/google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
   - Gradio: [https://gradio.app/docs/](https://gradio.app/docs/)
   - PyTorch: [https://pytorch.org/](https://pytorch.org/)
4. **Related Work**

   - Plant disease detection using deep learning
   - Agricultural computer vision applications
   - Transfer learning in medical/agricultural imaging

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas where contributions would be especially valuable:

- Additional data augmentation techniques
- Model architecture improvements
- Mobile deployment optimizations
- Documentation enhancements
- Bug fixes and performance improvements

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Acknowledgments

- **Makerere AI Lab** and **NaCRRI** for providing the Beans dataset
- **Hugging Face** for the Transformers library and model hub
- **Google Research** for the Vision Transformer architecture
- The open-source community for various tools and libraries

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub or contact the project maintainer:

- **GitHub**: [@Cemputus](https://github.com/Cemputus)
- **Repository**: [Deep-Learning-Project](https://github.com/Cemputus/Deep-Learning-Project)

---

**Note**: This project is part of academic research focused on applying deep learning to agricultural challenges. The models and code are provided for educational and research purposes.

---

<div align="center">

**Made for Agricultural Innovation**

[⭐ Star this repo](https://github.com/Cemputus/Deep-Learning-Project) | [🐛 Report Bug](https://github.com/Cemputus/Deep-Learning-Project/issues) | [💡 Request Feature](https://github.com/Cemputus/Deep-Learning-Project/issues)

</div>
