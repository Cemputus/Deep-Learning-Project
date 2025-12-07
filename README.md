# 🌱 Bean Leaf Disease Detection Using Vision Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cemputus/Deep-Learning-Project/blob/main/image_classification_vit_base_and_advanced.ipynb)

A deep learning solution for automated bean leaf disease detection using Vision Transformers (ViT). This project implements and compares two ViT models to classify bean leaves into three categories: Healthy, Angular Leaf Spot, and Bean Rust.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Sustainable Development Goals (SDGs)](#-sustainable-development-goals-sdgs)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Installation & Setup](#-installation--setup)
- [Usage & Running the Project](#-usage--running-the-project)
- [Troubleshooting](#-troubleshooting)
- [Evaluation Metrics](#-evaluation-metrics)
- [Gradio Demo](#-gradio-demo)
- [Key Findings](#-key-findings)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [References](#-references)
- [Contributing](#-contributing)
- [License](#-license)

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

## 📁 Project Structure

```
Deep-Learning-Project/
│
├── image_classification_vit_base_and_advanced.ipynb  # Main Jupyter notebook
├── README.md                                           # This file
├── requirements.txt                                    # Python dependencies
│
├── Deep_Learning_project_Report.pdf                    # Project report (PDF)
├── Deep Learning Presentation.pptx                     # Presentation slides
├── DEEP LEARNING POSTER PORTRAIT.pdf                  # Project poster
│
├── vit-baseline-beans/                                # Baseline model outputs (created during training)
│   ├── checkpoint-50/
│   ├── checkpoint-100/
│   ├── trainer_state.json
│   └── training_args.bin
│
├── vit-advanced-beans/                                 # Advanced model outputs (created during training)
│   ├── checkpoint-50/
│   ├── checkpoint-100/
│   ├── trainer_state.json
│   └── training_args.bin
│
└── cache/                                              # Dataset cache (created automatically)
    └── datasets/
        └── beans/
```

### File Descriptions

#### Core Files

- **`image_classification_vit_base_and_advanced.ipynb`**
  - Main project notebook containing all code
  - Organized into sections: Data Loading, EDA, Preprocessing, Training, Evaluation, Demo
  - Can be run in Google Colab or locally with Jupyter

- **`requirements.txt`**
  - Lists all Python package dependencies
  - Includes version specifications for compatibility
  - Use with `pip install -r requirements.txt`

- **`README.md`**
  - Comprehensive project documentation
  - Installation and usage instructions
  - Technical details and troubleshooting

#### Documentation Files

- **`Deep_Learning_project_Report.pdf`**
  - Detailed project report
  - Methodology, results, and analysis

- **`Deep Learning Presentation.pptx`**
  - Presentation slides
  - Project overview and key findings

- **`DEEP LEARNING POSTER PORTRAIT.pdf`**
  - Academic poster
  - Visual summary of the project

#### Generated Directories (Created During Execution)

- **`vit-baseline-beans/`**
  - Baseline model checkpoints
  - Training logs and metrics
  - Best model weights

- **`vit-advanced-beans/`**
  - Advanced model checkpoints
  - Training logs and metrics
  - Best model weights

- **`cache/`**
  - Hugging Face dataset cache
  - Automatically created when loading datasets
  - Can be deleted to re-download fresh data

### Notebook Structure

The notebook is organized into the following sections:

1. **Introduction & Setup** (Cells 1-5)
   - Project overview
   - Package installation
   - Environment setup

2. **Data Collection** (Cells 6-10)
   - Dataset loading from Hugging Face
   - Dataset structure examination

3. **Data Preprocessing** (Cells 11-40)
   - Image preprocessing
   - Data augmentation
   - Transform pipelines

4. **Exploratory Data Analysis** (Cells 41-50)
   - Class distribution
   - Image statistics
   - Visualizations

5. **Model Development** (Cells 51-70)
   - Baseline model creation
   - Advanced model loading
   - Model configuration

6. **Training** (Cells 71-100)
   - Training arguments setup
   - Baseline model training
   - Advanced model fine-tuning

7. **Evaluation** (Cells 101-120)
   - Test set evaluation
   - Metrics calculation
   - Confusion matrices
   - ROC curves

8. **Visualization** (Cells 121-140)
   - Training curves
   - Error analysis
   - Sample predictions

9. **Gradio Demo** (Cells 141-150)
   - Interactive web application
   - Model deployment

10. **Conclusion** (Cells 151+)
    - Summary
    - Key findings
    - Future work

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

## 🔬 Methodology & Technical Implementation

### Technical Stack

- **Deep Learning Framework**: PyTorch 2.0+
- **Transformer Library**: Hugging Face Transformers 4.30+
- **Data Processing**: Hugging Face Datasets, NumPy, Pandas
- **Image Processing**: Pillow (PIL), Torchvision
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: Scikit-learn, Hugging Face Evaluate
- **Deployment**: Gradio
- **Development**: Jupyter Notebook

### Data Preprocessing Pipeline

#### 1. Image Processing

**Step 1: Image Loading**
```python
from datasets import load_dataset
ds = load_dataset('beans')
# Images are loaded as PIL Image objects
```

**Step 2: ViT Image Processor**
```python
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained(
    'google/vit-base-patch16-224-in21k'
)

# Processing pipeline:
# 1. Resize to 224×224 (maintaining aspect ratio)
# 2. Convert to tensor
# 3. Normalize using ImageNet statistics:
#    - Mean: [0.485, 0.456, 0.406]
#    - Std: [0.229, 0.224, 0.225]
```

**Technical Details:**
- **Input Size**: 500×500 pixels (original)
- **Output Size**: 224×224 pixels (ViT standard)
- **Normalization**: Zero-mean, unit-variance using ImageNet statistics
- **Color Space**: RGB (3 channels)
- **Data Type**: Float32 tensors

#### 2. Data Augmentation Strategy

**Training Set Augmentations:**

```python
from torchvision import transforms

train_augmentation = transforms.Compose([
    # Geometric Transformations
    transforms.RandomHorizontalFlip(p=0.5),      # Horizontal mirroring
    transforms.RandomVerticalFlip(p=0.3),        # Vertical mirroring
    transforms.RandomRotation(degrees=30),       # Rotation ±30°
    
    # Color Transformations
    transforms.ColorJitter(
        brightness=0.2,    # ±20% brightness variation
        contrast=0.2,      # ±20% contrast variation
        saturation=0.2,    # ±20% saturation variation
        hue=0.1           # ±10% hue variation
    ),
    
    # Affine Transformations
    transforms.RandomAffine(
        degrees=0,                    # No rotation (already handled)
        translate=(0.1, 0.1),         # ±10% translation
        scale=(0.9, 1.1),             # 90-110% scaling
        shear=None                    # No shearing
    )
])
```

**Augmentation Rationale:**
- **Geometric Augmentations**: Handle different camera angles and orientations
- **Color Augmentations**: Account for varying lighting conditions
- **Affine Transformations**: Simulate different viewing perspectives

**Validation/Test Sets:**
- No augmentation applied (only preprocessing)
- Ensures fair evaluation on original images

#### 3. Data Splits

**Split Strategy:**
- **Training**: 80% (1,034 images) - Model learning
- **Validation**: 10% (133 images) - Hyperparameter tuning
- **Test**: 10% (128 images) - Final evaluation

**Class Distribution (Per Split):**
- Well-balanced across all splits (~33% per class)
- Prevents class imbalance issues
- Ensures representative evaluation

### Model Training Strategy

#### Baseline Model Training

**Architecture Configuration:**
```python
ViTConfig(
    image_size=224,
    patch_size=16,              # 14×14 patches per image
    num_channels=3,             # RGB
    hidden_size=384,            # Embedding dimension
    num_hidden_layers=6,       # Transformer blocks
    num_attention_heads=6,     # Multi-head attention
    intermediate_size=1536,     # FFN dimension
    hidden_dropout_prob=0.1,   # Dropout rate
    attention_probs_dropout_prob=0.1,
    num_labels=3               # Output classes
)
```

**Training Configuration:**
- **Optimizer**: AdamW (weight decay=0.01)
- **Learning Rate**: 5e-4 (linear warmup + cosine decay)
- **Batch Size**: 16 per device
- **Epochs**: 10
- **Gradient Clipping**: 1.0
- **Mixed Precision**: FP16 (when GPU available)

**Training Process:**
1. Initialize random weights (no pre-training)
2. Train for 10 epochs
3. Evaluate every 50 steps
4. Save best model based on validation accuracy
5. Total training steps: ~650

#### Advanced Model Fine-tuning

**Pre-trained Model:**
- **Base Model**: `google/vit-base-patch16-224-in21k`
- **Pre-training**: ImageNet-21k (14M images, 21,843 classes)
- **Architecture**: ViT-Base (12 layers, 768 hidden size)

**Fine-tuning Strategy:**
```python
# Load pre-trained model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=3  # Replace classification head
)

# Freeze backbone (optional - not done here)
# for param in model.vit.parameters():
#     param.requires_grad = False
```

**Fine-tuning Configuration:**
- **Learning Rate**: 2e-4 (lower than baseline - fine-tuning best practice)
- **Epochs**: 4 (fewer needed due to pre-training)
- **Batch Size**: 16
- **Warmup Steps**: 50
- **Weight Decay**: 0.01

**Why Transfer Learning Works:**
1. Pre-trained features capture general visual patterns
2. Fine-tuning adapts these features to bean leaf diseases
3. Requires less data and training time
4. Better generalization to new images

### Training Infrastructure

#### Data Collator

```python
def collate_fn(batch):
    """Batch data for training"""
    return {
        'pixel_values': torch.stack([
            x['pixel_values'].squeeze(0) for x in batch
        ]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }
```

#### Evaluation Metrics

```python
def compute_metrics(p):
    """Compute classification metrics"""
    predictions = np.argmax(p.predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(p.label_ids, predictions),
        'precision': precision_score(p.label_ids, predictions, average='weighted'),
        'recall': recall_score(p.label_ids, predictions, average='weighted'),
        'f1': f1_score(p.label_ids, predictions, average='weighted')
    }
```

#### Checkpointing Strategy

- **Save Frequency**: Every 50 steps
- **Best Model**: Based on validation accuracy
- **Total Checkpoints**: Limited to 2 (save_total_limit=2)
- **Checkpoint Contents**:
  - Model weights
  - Optimizer state
  - Training arguments
  - Training metrics

### Computational Requirements

#### Memory Usage

- **Baseline Model**: ~2GB GPU memory (batch size 16)
- **Advanced Model**: ~4GB GPU memory (batch size 16)
- **CPU Memory**: ~8GB RAM minimum

#### Training Time (Approximate)

| Model | Hardware | Time |
|-------|----------|------|
| Baseline | GPU (T4) | 30-60 min |
| Baseline | CPU (8 cores) | 2-4 hours |
| Advanced | GPU (T4) | 15-30 min |
| Advanced | CPU (8 cores) | 1-2 hours |

#### Inference Time

- **Single Image (GPU)**: ~10-20ms
- **Single Image (CPU)**: ~100-200ms
- **Batch (16 images, GPU)**: ~50-100ms

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

## 🚀 Installation & Setup

### System Requirements

#### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 8GB minimum (16GB recommended for training)
- **Storage**: 10GB free disk space (20GB+ recommended for model checkpoints)
- **CPU**: Multi-core processor (4+ cores recommended)

#### Recommended Requirements (for Training)
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
  - CUDA 11.8 or 12.1+
  - cuDNN 8.0+
- **RAM**: 16GB or more
- **Storage**: SSD with 50GB+ free space

#### For Inference Only
- **CPU**: Modern multi-core processor
- **RAM**: 4GB minimum
- **Storage**: 5GB free space

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Cemputus/Deep-Learning-Project.git

# Navigate to the project directory
cd Deep-Learning-Project

# Verify you're in the correct directory
ls -la  # On Linux/Mac
dir     # On Windows
```

**Expected Output:**
```
image_classification_vit_base_and_advanced.ipynb
README.md
requirements.txt
Deep_Learning_project_Report.pdf
...
```

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies and prevents conflicts with other projects.

#### Option A: Using venv (Python 3.8+)

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (you should see (venv) in your prompt)
```

**Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in your prompt)
```

#### Option B: Using conda (Alternative)

```bash
# Create conda environment
conda create -n bean-disease-detection python=3.9 -y

# Activate environment
conda activate bean-disease-detection
```

### Step 3: Install Dependencies

#### Method 1: Using requirements.txt (Recommended)

```bash
# Ensure pip is up to date
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -E "torch|transformers|datasets"
```

**Expected Output:**
```
torch                   2.x.x
torchvision             0.x.x
transformers            4.x.x
datasets                2.x.x
...
```

#### Method 2: Manual Installation

If you encounter issues with `requirements.txt`, install packages individually:

```bash
# Core deep learning
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# OR for CPU only:
pip install torch torchvision

# Hugging Face ecosystem
pip install transformers>=4.30.0
pip install datasets>=2.12.0
pip install accelerate>=0.20.0
pip install evaluate>=0.4.0

# Image processing
pip install Pillow>=9.5.0

# Data processing
pip install numpy>=1.24.0 pandas>=2.0.0

# Machine learning
pip install scikit-learn>=1.3.0

# Visualization
pip install matplotlib>=3.7.0 seaborn>=0.12.0

# Interactive demo
pip install gradio>=3.35.0

# Jupyter support
pip install jupyter>=1.0.0 ipykernel>=6.25.0 notebook>=6.5.0

# Optional but recommended
pip install tensorboard>=2.13.0 tqdm>=4.65.0
```

### Step 4: Verify Installation

Create a verification script to test your installation:

```python
# test_installation.py
import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets: {e}")
        return False
    
    try:
        import PIL
        print(f"✅ Pillow {PIL.__version__}")
    except ImportError as e:
        print(f"❌ Pillow: {e}")
        return False
    
    try:
        import numpy, pandas, sklearn, matplotlib, seaborn, gradio
        print("✅ All other dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    print("\n✅ All dependencies installed successfully!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
```

Run the verification:

```bash
python test_installation.py
```

**Expected Output:**
```
Testing imports...
✅ PyTorch 2.x.x
   CUDA available: True
   CUDA version: 11.8
   GPU: NVIDIA GeForce RTX 3080
✅ Transformers 4.x.x
✅ Datasets 2.x.x
✅ Pillow 10.x.x
✅ All other dependencies installed

✅ All dependencies installed successfully!
```

### Step 5: Configure GPU (Optional but Recommended)

If you have an NVIDIA GPU, verify CUDA setup:

```python
# check_cuda.py
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  CUDA not available. Training will use CPU (much slower).")
```

Run:
```bash
python check_cuda.py
```

## 💻 Usage & Running the Project

This section provides comprehensive step-by-step instructions for running the project in different environments.

### 🚀 Quick Start (5-Minute Setup)

For users who want to get started quickly:

```bash
# 1. Clone repository
git clone https://github.com/Cemputus/Deep-Learning-Project.git
cd Deep-Learning-Project

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Jupyter Notebook
jupyter notebook

# 5. Open image_classification_vit_base_and_advanced.ipynb
# 6. Run all cells (Cell → Run All)
```

**Expected Time:**
- Installation: 5-10 minutes
- Full training: 30-60 minutes (GPU) or 2-4 hours (CPU)

### Detailed Step-by-Step Guide

This section provides comprehensive step-by-step instructions for running the project in different environments.

### Option 1: Running in Google Colab (Recommended for Beginners)

Google Colab provides free GPU access and requires no local setup.

#### Step 1: Open the Notebook

1. Click the **"Open In Colab"** badge at the top of this README
2. Or visit: `https://colab.research.google.com/github/Cemputus/Deep-Learning-Project/blob/main/image_classification_vit_base_and_advanced.ipynb`

#### Step 2: Enable GPU Runtime

1. In Colab, go to **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4, V100, or A100)
3. Click **Save**

#### Step 3: Run the Notebook

1. The notebook is organized into cells. Run cells sequentially:
   - Click on a cell
   - Press `Shift + Enter` to run the cell
   - Or use **Runtime** → **Run all** to execute all cells

#### Step 4: Monitor Training Progress

- Training progress will be displayed in the notebook cells
- Baseline model: ~30-60 minutes on GPU
- Advanced model: ~15-30 minutes on GPU
- Monitor GPU usage: **Runtime** → **Manage sessions**

### Option 2: Running Locally with Jupyter Notebook

#### Step 1: Start Jupyter Notebook Server

```bash
# Activate your virtual environment first
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Start Jupyter Notebook
jupyter notebook

# Alternative: Use JupyterLab
jupyter lab
```

This will open Jupyter in your default web browser (usually at `http://localhost:8888`).

#### Step 2: Open the Notebook

1. Navigate to the project directory in Jupyter
2. Click on `image_classification_vit_base_and_advanced.ipynb`
3. The notebook will open in a new tab

#### Step 3: Select Kernel

1. Go to **Kernel** → **Change Kernel**
2. Select your virtual environment (e.g., `venv` or `bean-disease-detection`)

#### Step 4: Run Cells

- **Run single cell**: `Shift + Enter`
- **Run cell and move to next**: `Shift + Enter`
- **Run all cells**: **Cell** → **Run All**
- **Run all above**: **Cell** → **Run All Above**

### Option 3: Running as Python Script (Advanced)

You can convert the notebook to a Python script for command-line execution:

```bash
# Convert notebook to script
jupyter nbconvert --to script image_classification_vit_base_and_advanced.ipynb

# Run the script
python image_classification_vit_base_and_advanced.py
```

### Detailed Workflow: Step-by-Step Execution

The notebook follows this structured workflow:

#### Phase 1: Environment Setup & Data Loading

**Cells 1-5: Project Introduction & Package Installation**

```python
# These cells will:
# 1. Install required packages (if not already installed)
# 2. Display project overview
# 3. Set up the environment
```

**Expected Output:**
```
Checking and installing required packages...
✅ Package installation check complete!
```

**Cells 6-10: Dataset Loading**

```python
from datasets import load_dataset

# Load the beans dataset
ds = load_dataset('beans')
print(f"Dataset splits: {list(ds.keys())}")
```

**Expected Output:**
```
Loading beans dataset from Hugging Face...
Dataset loaded successfully!
Dataset splits: ['train', 'validation', 'test']
```

**Action Items:**
- ✅ Verify dataset loads successfully
- ✅ Check dataset statistics (1,034 train, 133 validation, 128 test)
- ✅ Review sample images from each class

#### Phase 2: Exploratory Data Analysis (EDA)

**Cells 11-30: Data Analysis**

This phase includes:
- Class distribution analysis
- Image statistics (dimensions, color distributions)
- Sample image visualization
- Data quality checks

**Key Metrics to Verify:**
- Class balance: ~33% per class
- Image dimensions: 500×500 pixels
- Total images: 1,295

#### Phase 3: Data Preprocessing

**Cells 31-40: Preprocessing Setup**

```python
from transformers import ViTImageProcessor

# Load image processors
image_processor_advanced = ViTImageProcessor.from_pretrained(
    'google/vit-base-patch16-224-in21k'
)
image_processor_baseline = ViTImageProcessor.from_pretrained(
    'google/vit-base-patch16-224'
)
```

**Action Items:**
- ✅ Verify processors load correctly
- ✅ Test preprocessing on sample image
- ✅ Confirm image size: 224×224 after preprocessing

**Cells 41-50: Data Augmentation**

```python
from torchvision import transforms

# Define augmentation pipeline
train_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    # ... more augmentations
])
```

#### Phase 4: Model Development

**Cells 51-60: Baseline Model Creation**

```python
from transformers import ViTForImageClassification, ViTConfig

# Create custom ViT configuration
baseline_config = ViTConfig(
    image_size=224,
    patch_size=16,
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=6,
    num_labels=3
)

# Initialize model
baseline_model = ViTForImageClassification(baseline_config)
```

**Expected Output:**
```
Baseline ViT model created!
Model parameters: 11,020,035
```

**Cells 61-70: Advanced Model Creation**

```python
# Load pre-trained model
advanced_model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=3
)
```

**Expected Output:**
```
Advanced ViT model (pre-trained) loaded!
Model parameters: 85,800,963
```

#### Phase 5: Training Configuration

**Cells 71-80: Training Arguments**

```python
from transformers import TrainingArguments

baseline_training_args = TrainingArguments(
    output_dir="./vit-baseline-beans",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=5e-4,
    # ... more arguments
)
```

**Important Parameters:**
- **Batch Size**: 16 (reduce to 8 if GPU memory is limited)
- **Learning Rate**: 5e-4 (baseline), 2e-4 (advanced)
- **Epochs**: 10 (baseline), 4 (advanced)
- **Evaluation**: Every 50 steps

#### Phase 6: Model Training

**⚠️ Training Time Estimates:**
- **Baseline Model (GPU)**: 30-60 minutes
- **Baseline Model (CPU)**: 2-4 hours
- **Advanced Model (GPU)**: 15-30 minutes
- **Advanced Model (CPU)**: 1-2 hours

**Cells 81-90: Train Baseline Model**

```python
from transformers import Trainer

baseline_trainer = Trainer(
    model=baseline_model,
    args=baseline_training_args,
    train_dataset=ds_baseline["train"],
    eval_dataset=ds_baseline["validation"],
    # ... more arguments
)

# Start training
baseline_train_results = baseline_trainer.train()
```

**Monitoring Training:**
- Watch for loss decreasing over epochs
- Check validation accuracy improving
- Monitor GPU/CPU usage
- Check for out-of-memory errors

**Expected Training Output:**
```
***** Running training *****
  Num examples = 1034
  Num Epochs = 10
  Instantaneous batch size per device = 16
  Total train batch size = 16
  Gradient Accumulation steps = 1
  Total optimization steps = 650
  
Epoch 1/10: 100%|████████| 65/65 [02:15<00:00,  2.08s/it]
  Train loss: 0.9234
  Eval accuracy: 0.6541
  ...
```

**Cells 91-100: Train Advanced Model**

```python
advanced_trainer = Trainer(
    model=advanced_model,
    args=advanced_training_args,
    train_dataset=ds_advanced["train"],
    eval_dataset=ds_advanced["validation"],
    # ... more arguments
)

# Start fine-tuning
advanced_train_results = advanced_trainer.train()
```

**Save Models:**
Models are automatically saved to:
- `./vit-baseline-beans/` (baseline model)
- `./vit-advanced-beans/` (advanced model)

#### Phase 7: Model Evaluation

**Cells 101-110: Test Set Evaluation**

```python
# Evaluate baseline model
baseline_test_metrics = baseline_trainer.evaluate(ds_baseline['test'])

# Evaluate advanced model
advanced_test_metrics = advanced_trainer.evaluate(ds_advanced['test'])
```

**Expected Metrics:**
- Baseline: ~75% accuracy
- Advanced: ~95% accuracy

**Cells 111-120: Generate Visualizations**

- Confusion matrices
- ROC curves
- Precision-Recall curves
- Training curves
- Error analysis

#### Phase 8: Gradio Demo

**Cells 121-130: Launch Interactive Demo**

```python
import gradio as gr

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(), gr.Number(), gr.Label()],
    title="🌱 Bean Leaf Disease Detector"
)

# Launch demo
demo.launch(share=True)  # Creates public URL
```

**Access the Demo:**
- Local URL: `http://127.0.0.1:7860`
- Public URL: Provided by Gradio (expires in 72 hours)

### Quick Start: Inference Only (Using Pre-trained Models)

If you only want to use pre-trained models for inference without training:

#### Step 1: Download Pre-trained Models

```bash
# Create models directory
mkdir -p models/baseline models/advanced

# Download models (if available)
# Note: You may need to train models first or download from model hub
```

#### Step 2: Create Inference Script

Create `inference.py`:

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Configuration
MODEL_PATH = "path/to/your/model"  # Update this
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']

def load_model(model_path):
    """Load trained model and processor"""
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(
        'google/vit-base-patch16-224-in21k'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, processor, device

def predict(image_path, model, processor, device):
    """Make prediction on a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt").to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()
    
    return CLASS_NAMES[pred_id], confidence

if __name__ == "__main__":
    # Load model
    model, processor, device = load_model(MODEL_PATH)
    
    # Example prediction
    image_path = "path/to/bean_leaf.jpg"
    prediction, confidence = predict(image_path, model, processor, device)
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")
```

#### Step 3: Run Inference

```bash
python inference.py
```

### Command-Line Interface (CLI) Usage

For batch processing, create a CLI script:

```python
# cli_predict.py
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Bean Leaf Disease Detection'
    )
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='./vit-advanced-beans',
        help='Path to model directory'
    )
    parser.add_argument(
        '--batch', 
        type=str,
        help='Path to directory of images for batch processing'
    )
    
    args = parser.parse_args()
    
    # Load model and make predictions
    # ... (implementation)
    
if __name__ == "__main__":
    main()
```

Usage:
```bash
# Single image
python cli_predict.py --image path/to/image.jpg

# Batch processing
python cli_predict.py --batch path/to/images/ --model ./vit-advanced-beans
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

1. **Reduce Batch Size:**
   ```python
   # In TrainingArguments
   per_device_train_batch_size=8  # Reduce from 16 to 8
   per_device_eval_batch_size=8
   ```

2. **Enable Gradient Accumulation:**
   ```python
   gradient_accumulation_steps=2  # Effectively doubles batch size
   ```

3. **Use Mixed Precision:**
   ```python
   fp16=True  # Already enabled, but verify
   ```

4. **Clear GPU Cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

#### Issue 2: Package Installation Errors

**Error:** `ERROR: Could not find a version that satisfies the requirement`

**Solutions:**

1. **Update pip:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Install from specific index:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install packages individually:**
   ```bash
   pip install transformers
   pip install datasets
   # ... etc
   ```

#### Issue 3: Dataset Download Fails

**Error:** `ConnectionError` or `TimeoutError` when loading dataset

**Solutions:**

1. **Use Hugging Face Cache:**
   ```python
   from datasets import load_dataset
   ds = load_dataset('beans', cache_dir='./cache')
   ```

2. **Set Hugging Face Token (if required):**
   ```bash
   export HF_TOKEN=your_token_here
   ```

3. **Retry with timeout:**
   ```python
   import os
   os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
   ```

#### Issue 4: Jupyter Kernel Not Found

**Error:** Kernel not found when running notebook

**Solutions:**

1. **Install ipykernel:**
   ```bash
   pip install ipykernel
   ```

2. **Register kernel:**
   ```bash
   python -m ipykernel install --user --name=bean-disease-detection
   ```

3. **Select kernel in Jupyter:**
   - Kernel → Change Kernel → bean-disease-detection

#### Issue 5: Model Training Very Slow

**Symptoms:** Training takes hours even on GPU

**Solutions:**

1. **Verify GPU Usage:**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

2. **Check Data Loading:**
   ```python
   # Use num_workers for data loading
   from torch.utils.data import DataLoader
   dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)
   ```

3. **Enable FP16:**
   ```python
   fp16=True  # In TrainingArguments
   ```

#### Issue 6: Import Errors

**Error:** `ModuleNotFoundError: No module named 'X'`

**Solutions:**

1. **Verify virtual environment:**
   ```bash
   which python  # Linux/Mac
   where python  # Windows
   ```

2. **Reinstall package:**
   ```bash
   pip uninstall package_name
   pip install package_name
   ```

3. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   ```

### Performance Optimization Tips

#### For Faster Training:

1. **Use GPU:** Training is 10-20x faster on GPU
2. **Increase Batch Size:** If GPU memory allows
3. **Use Mixed Precision:** FP16 reduces memory and speeds up training
4. **Optimize Data Loading:** Use `num_workers` and `pin_memory`
5. **Reduce Epochs:** Advanced model needs only 4 epochs

#### For Faster Inference:

1. **Use GPU:** Even for inference
2. **Batch Processing:** Process multiple images at once
3. **Model Quantization:** Reduce model size and speed
4. **ONNX Export:** Convert to ONNX for faster inference

### Getting Help

If you encounter issues not covered here:

1. **Check GitHub Issues:** [Project Issues](https://github.com/Cemputus/Deep-Learning-Project/issues)
2. **Create New Issue:** Include:
   - Error message (full traceback)
   - System information (OS, Python version, GPU)
   - Steps to reproduce
   - Expected vs actual behavior
3. **Check Documentation:**
   - [Hugging Face Docs](https://huggingface.co/docs/transformers)
   - [PyTorch Docs](https://pytorch.org/docs/stable/index.html)

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
