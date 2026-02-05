Image Classification Using CNNs and Transfer Learning Overview

This project explores deep learning approaches for multi-class image classification using a small, balanced colour image dataset. A custom Convolutional Neural Network (CNN) is developed and compared 
with a transfer learning approach using MobileNetV3Small to evaluate performance, efficiency, and generalisation. The focus of this work is on model design, hyperparameter tuning, and systematic evaluation, rather than achieving state-of-the-art accuracy.

Methods
The following approaches are implemented and compared:
Custom CNN
  Multiple convolution and pooling layers
  Regularisation using dropout and batch normalisation
  Hyperparameter tuning using Keras Tuner
Transfer Learning
  Pretrained MobileNetV3Small as the base model
  Frozen backbone with custom classification head
  Fine-tuning and evaluation on the target dataset

Evaluation
  Model performance is assessed using:
  Accuracy
  Precision, Recall, and F1-score (per class)
  Confusion matrices
  Training and validation learning curves

Comparative analysis highlights strengths and limitations of each approach when applied to small datasets.

Technologies Used
  Python
  TensorFlow / Keras
  Keras Tuner
  NumPy
  scikit-learn
  Google Colab

Repository Contents
– Main notebook containing model implementation, training, and evaluation

Notes

This repository is intended for learning and demonstration purposes and showcases experimentation with deep learning techniques for image classification.
