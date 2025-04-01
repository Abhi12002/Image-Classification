
# 🐶🐱 Image Classification: Dogs vs. Cats

This project demonstrates an end-to-end image classification pipeline using TensorFlow and Keras to distinguish between images of dogs and cats. The dataset is sourced from Kaggle and the model is trained with CNN (Convolutional Neural Networks) architecture.

---

## 📂 Dataset

- **Source:** [Kaggle - Dogs vs. Cats](https://www.kaggle.com/salader/dogs-vs-cats)
- The dataset contains two categories:
  - `dogs`
  - `cats`
- Images are split into training and validation sets using `image_dataset_from_directory`.

---

## 🛠️ Features

- Dataset preprocessing and normalization
- Image augmentation support
- CNN architecture with:
  - Convolutional layers
  - Batch Normalization
  - Dropout for regularization
  - MaxPooling
- Model evaluation with accuracy and loss plots

---

## 🚀 Getting Started

### Prerequisites

Install dependencies with:

```bash
pip install tensorflow kaggle matplotlib
```

Ensure your `kaggle.json` API token is placed properly:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-classification-dogs-vs-cats.git
   cd image-classification-dogs-vs-cats
   ```

2. Launch Jupyter Notebook or Colab and open `Image Classification.ipynb`.

3. Run all cells to:
   - Download the dataset
   - Preprocess images
   - Train and evaluate the model

---

## 📊 Model Performance

The notebook provides plots for training/validation accuracy and loss. These visuals help track overfitting and convergence trends.

---

## 📁 Project Structure

```
.
├── Image Classification.ipynb
├── kaggle.json
├── /content
│   ├── train/
│   └── test/
└── README.md
```

---

## 📌 TODOs

- [ ] Add early stopping and learning rate scheduler
- [ ] Export trained model (.h5 or SavedModel format)
- [ ] Include image augmentation pipeline
- [ ] Add confusion matrix and classification report

---

## 🧠 Credits

- Built with TensorFlow and Keras
- Dataset courtesy of Kaggle
- Developed by Abhinav Mishra (https://github.com/Abhi12002)
