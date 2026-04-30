# 🌿 Plant Disease Detection Using Deep Learning

> **Senior Design Project** · VIT-AP University · Dec 2024  
> **Team:** Satyala Murali Karthik · **Mekala Samuel** · Kurmala Bhanu Prakash  
> **Guide:** Dr. S. Kalyani · School of Computer Science & Engineering

---

## 📌 Overview

Plant diseases are a major threat to global food security. This project builds an **automated deep learning system** that classifies plant leaf images into **38 healthy and diseased categories** using the <a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset" target="_blank">New Plant Diseases Dataset</a> (87,000+ images from Kaggle).

We implemented and **benchmarked 7 CNN architectures** side-by-side using transfer learning, and integrated the best model into a **real-time GUI** for practical agricultural use.

---

## 🏆 Results

| Model | Accuracy | F1 Score | ROC AUC |
|---|---|---|---|
| **GoogleNet (Ours)** | **99.10%** | **99.00%** | **99.00%** |
| DenseNet | ~98.5% | ~98.3% | ~98.4% |
| ResNet-50 | ~97.8% | ~97.5% | ~97.6% |
| VGG19 | ~97.2% | ~96.9% | ~97.0% |
| VGG16 | ~96.5% | ~96.2% | ~96.3% |
| AlexNet | ~94.1% | ~93.8% | ~93.9% |
| LeNet-5 | ~85.0% | ~84.5% | ~84.7% |

> **GoogleNet achieved the highest accuracy of 99.1%** — selected as the final production model.

---

## 🧠 Models Implemented

| Model | Architecture Highlights | Use Case |
|---|---|---|
| **LeNet-5** | 2 conv + 2 FC layers · 32×32 input | Lightweight baseline |
| **AlexNet** | 5 conv + 3 FC · ReLU · Dropout | Fast inference |
| **VGG16** | 13 conv + 3 dense · 224×224 | Feature-rich classification |
| **VGG19** | 16 conv + 3 dense · deeper VGG | High spatial detail |
| **ResNet-50** | 50 layers · Skip connections | Avoids vanishing gradients |
| **GoogleNet** | Inception modules · 22 layers | ✅ Best trade-off: accuracy + speed |
| **DenseNet** | Dense connectivity · feature reuse | Near-best with compact params |

---

## ⚙️ Methodology

### Dataset
- **87,000+ images** across **38 classes** (plant species × disease/healthy states)
- Format: JPEG · Varied resolutions

### Preprocessing Pipeline
```
Raw Images
    → Resize (224×224 for most; 32×32 for LeNet-5)
    → Normalize (ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    → Augment (H/V Flip · Rotation ±20° · Random Crop · Color Jitter)
    → Tensor Conversion
    → Split (70% Train / 15% Val / 15% Test)
```

### Training Configuration
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001) / SGD with momentum
- **Scheduler:** StepLR (step_size=7, γ=0.1)
- **Regularization:** Dropout + Early Stopping (patience=3)
- **Max Epochs:** 20

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-blue?style=flat-square)

**Hardware:** GPU-enabled (NVIDIA CUDA) · 8GB+ RAM

---

## 🖥️ GUI

A **Tkinter-based desktop GUI** was built for real-time leaf disease classification:
- Upload any leaf image via file dialog
- Instant disease class prediction with confidence score
- User-friendly interface for farmers and agricultural experts

---

## 📁 Project Structure

```
plant-disease-detection/
├── alexnet.py          # AlexNet implementation
├── googlenet.py        # GoogleNet (Inception V1) — best model (99.1%)
├── models.py           # VGG16, VGG19, ResNet50, LeNet-5, DenseNet
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/samuel-mekala/plant-disease-detection.git
cd plant-disease-detection

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place in ./data/
# https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

# Train GoogleNet (best model)
python googlenet.py

# Train AlexNet
python alexnet.py

# Train other models (VGG16, VGG19, ResNet50, LeNet-5, DenseNet)
python models.py --model vgg16
python models.py --model vgg19
python models.py --model resnet50
python models.py --model lenet5
python models.py --model densenet
```

---

## 🔮 Future Work

- [ ] Ensemble learning combining GoogleNet + DenseNet
- [ ] Vision Transformers (ViT) for spatial attention
- [ ] Mobile deployment via TensorFlow Lite / ONNX
- [ ] Explainable AI with Grad-CAM visualizations
- [ ] Multilingual voice-based interface for rural farmers

---

## 📚 References

1. Mohanty et al. (2016) — AlexNet & GoogLeNet on PlantVillage dataset
2. Ferentinos (2018) — VGG & ResNet transfer learning for plant disease
3. He et al. (2015) — Deep Residual Learning (ResNet)
4. LeCun et al. (2015) — Deep Learning foundations

---

*VIT-AP University · Computer Science & Engineering · Dec 2024*
