<h1 align="center">🐾 Image Classification: Dogs vs. Cats</h1>

<p align="center">
  <em>State-of-the-art CNN for classifying images as cats or dogs, built with TensorFlow & Keras.</em>
</p>

<p align="center">
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/Framework-TensorFlow-orange.svg"></a>
  <a href="https://keras.io/"><img src="https://img.shields.io/badge/Keras-API-red.svg"></a>
  <a href="https://github.com/Abhi12002/Image-Classification/stargazers"><img src="https://img.shields.io/github/stars/Abhi12002/Image-Classification?style=social"></a>
</p>

---

## 📑 Table of Contents

* [📝 Project Overview](#-project-overview)
* [📊 Dataset](#-dataset)
* [🧠 Model Architecture](#-model-architecture)
* [🧹 Data Preprocessing](#-data-preprocessing)
* [🚀 Training & Evaluation](#-training--evaluation)
* [📈 Results & Visualization](#-results--visualization)
* [📁 Project Structure](#-project-structure)
* [⚙️ Installation & Setup](#-installation--setup)
* [🔬 Experiments & Benchmarks](#-experiments--benchmarks)
* [🛣️ Future Work](#-future-work)
* [📜 License](#-license)
* [🙏 Acknowledgments](#-acknowledgments)
* [📬 Contact](#-contact)

---

## 📝 Project Overview

This repository demonstrates how to build a robust Convolutional Neural Network (CNN) for classifying images as either cats or dogs, leveraging the Kaggle Dogs vs. Cats dataset. The project is designed for both learning and practical application, following best practices in data science and open-source ML development.

---

## 📊 Dataset

* **Source**: [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
* **Classes**: `dog`, `cat`
* **Size**: \~25,000 labeled images
* **Splits**: Training, Validation, Test

---

## 🧠 Model Architecture

* **Framework**: Keras Sequential API (TensorFlow backend)
* **Layers**:

  * Multiple Conv2D layers with ReLU activation
  * MaxPooling for spatial downsampling
  * Batch Normalization for stable training
  * Dropout for regularization
  * Dense output with sigmoid activation

---

## 🧹 Data Preprocessing

* Image resizing and normalization
* Efficient directory-based loading with `image_dataset_from_directory`
* On-the-fly data augmentation:

  * Random flips
  * Rotations
  * Zooms

---

## 🚀 Training & Evaluation

* **Loss**: Binary Crossentropy
* **Optimizer**: Adam
* **Metrics**: Accuracy
* **Epochs**: 10–20 (configurable)
* **Validation**: Early stopping, learning rate scheduling

---

## 📈 Results & Visualization

* **Training Accuracy**: \~95%
* **Validation Accuracy**: \~93%

**Performance Highlights:**

* Stable generalization, minimal overfitting
* Visual sample predictions included
* Confusion matrix and classification report

<details>
<summary>Sample Output</summary>

| Metric         | Value |
| -------------- | ----- |
| Train Accuracy | 95%   |
| Val Accuracy   | 93%   |

</details>

---

## 📁 Project Structure

```
Image-Classification/
├── Image Classification.ipynb
├── /data/                # Training and validation images
├── /models/              # Saved model artifacts
├── /outputs/             # Plots, confusion matrix, sample predictions
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/Abhi12002/Image-Classification.git
cd Image-Classification
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
# Or manually:
pip install tensorflow keras matplotlib
```

3. **Run the notebook**

```bash
jupyter notebook "Image Classification.ipynb"
```

---

## 🔬 Experiments & Benchmarks

* Baseline CNN vs. pretrained models (e.g., ResNet, MobileNet)
* Impact of advanced augmentation
* Training curves and confusion matrices for reproducibility

---

## 🛣️ Future Work

* Integrate pretrained models for improved accuracy
* Expand to multiclass animal classification
* Deploy as a web app (Flask/Streamlit)
* Add model interpretability (Grad-CAM, SHAP)

---

## 📜 License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## 🙏 Acknowledgments

* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
* Open-source ML community for ongoing inspiration and support

---

## 📬 Contact

**Abhinav Mishra**
[LinkedIn](https://www.linkedin.com/in/abhinav-mishra-4b72b120b/)
[GitHub](https://github.com/Abhi12002)

> ⭐ If this project helped you, please star the repo and share your feedback!
