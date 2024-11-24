# X-Ray Disease Classification Using Convolutional Neural Networks (CNN)

This project involves building a Convolutional Neural Network (CNN) model to classify diseases from X-ray images. It demonstrates the application of deep learning in medical imaging, using TensorFlow and Keras for model development and training.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Setup and Usage](#setup-and-usage)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Overview
The goal of this project is to classify X-ray images into categories (e.g., **Normal** and **Diseased**) using a deep learning model. The model was trained on a dataset containing X-ray images of patients and evaluated on unseen test data.

---

## Dataset
The dataset consists of X-ray images organized into three directories:
- **Train**: Used for training the model
- **Validation**: Used for tuning hyperparameters
- **Test**: Used for final evaluation

Directory structure:
```
dataset/
├── train/
│   ├── normal/
│   ├── diseased/
├── val/
│   ├── normal/
│   ├── diseased/
├── test/
    ├── normal/
    ├── diseased/
```

The images were preprocessed to a target size of **150x150 pixels**.

---

## Model Architecture
The CNN architecture includes:
- **Input layer**: Processes X-ray images resized to 150x150x3.
- **Convolutional layers**: Extract features using filters.
- **Max-pooling layers**: Reduce spatial dimensions.
- **Fully connected layers**: Learn complex patterns for classification.
- **Output layer**: Binary classification with `sigmoid` activation.

---

## Dependencies
To run this project, install the following dependencies:
- Python 3.8+
- TensorFlow 2.11+
- NumPy
- Pandas
- Matplotlib

Install the required libraries using:
```bash
pip install tensorflow numpy pandas matplotlib
```

---

## Setup and Usage

1. **Extract Dataset**  
   Download and extract the dataset. Organize it into `train`, `val`, and `test` directories as specified in the dataset structure.

2. **Train the Model**  
   Run the Python script to train the model:
   ```bash
   python train_model.py
   ```

3. **Save the Model**  
   The trained model is saved as a `.keras` file:
   ```python
   model.save('xray_classification_model.keras')
   ```

4. **Evaluate the Model**  
   Evaluate the model using the test dataset:
   ```bash
   python evaluate_model.py
   ```

---

## Results
- **Training Accuracy**: 95%
- **Validation Accuracy**: 93%
- **Test Accuracy**: 90%


## Future Work
- Improve accuracy with more advanced architectures like ResNet or VGG.
- Add more categories for multi-class classification.
- Implement Grad-CAM for visualizing feature maps.

---

## Acknowledgments
- Dataset:https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Tools: TensorFlow, Keras, Matplotlib
