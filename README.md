# Digit Recognition with CNN (DevifyX Internship Assignment)

This repository contains my internship assignment for DevifyX. It is a full end-to-end handwritten digit recognition project using the MNIST dataset, built with TensorFlow and Keras. The solution meets all mandatory requirements and includes multiple bonus features to demonstrate model robustness, interpretability, and deployment readiness.

---

## Project Overview

The goal of this project is to develop a robust Convolutional Neural Network (CNN) that can accurately classify handwritten digits from the MNIST dataset. The project demonstrates:
- Data preprocessing, augmentation, and visualization
- Model building with Batch Normalization and Dropout
- Hyperparameter tuning using Keras Tuner
- Early stopping and checkpointing to avoid overfitting
- Saving the final model in `.h5` and `.tflite` formats
- Inference on individual images
- Intermediate activation visualization to interpret learned features
- FGSM adversarial attack to test model vulnerability
- Simple adversarial training as a defense mechanism to improve robustness

---

## Results Summary

- Clean test accuracy: ~98%
- Adversarial (FGSM) test accuracy before defense: ~56%
- Adversarial (FGSM) test accuracy after defense: ~72%

This shows how adversarial examples can significantly reduce performance and how simple defenses like adversarial training can improve robustness.

---
## Project Screenshots

### Dataset Samples
![image alt](https://github.com/rira2/DevifyX-/blob/main/Assets/dataset%20pictures.png)

---

### Accuracy and Loss During Training

#### Accuracy Over Epochs
![image alt](https://github.com/rira2/DevifyX-/blob/main/Assets/accuracy%20over%20epochs.png)

#### Loss Over Epochs
![image alt](https://github.com/rira2/DevifyX-/blob/main/Assets/loss%20over%20epochs%20.png)

#### History
![History](Assets/history)

#### History Continued
![image alt](https://github.com/rira2/DevifyX-/blob/main/Assets/history%20cont..png)

---

### Model Evaluation

#### Confusion Matrix
![image alt](https://github.com/rira2/DevifyX-/blob/main/Assets/confusion%20matrix%20.png)

#### Classification Report
![image alt](https://github.com/rira2/DevifyX-/blob/main/Assets/classification%20report.png)

#### Model Summary
![image alt](https://github.com/rira2/DevifyX-/blob/main/Assets/model%20summary.png)

---

### Robustness & Adversarial Testing

#### Accuracy on Clean Test Data AFTER Adversarial Training
![image alt](https://github.com/rira2/DevifyX-/blob/main/Assets/Accuracy%20on%20clean%20test%20data%20AFTER%20adversarial%20training.png)

#### Accuracy on Adversarial Examples AFTER Defense
![image alt](https://github.com/rira2/DevifyX-/blob/main/Assets/Accuracy%20on%20adversarial%20examples%20AFTER%20defense%20.png))

#### Accuracy on Adversarial Examples AFTER Defense, Adverse Test Set
![image alt](Ahttps://github.com/rira2/DevifyX-/blob/main/Assets/Accuracy%20on%20adversarial%20examples%20AFTER%20defense%2Cadverse%20test%20set.png)

#### Accuracy After Adversarial Training
![image alt](https://github.com/rira2/DevifyX-/blob/main/Assets/acurracy%20after%20adversarial%20training.png)

#### Loss After Adversarial Training
![image alt](Ahttps://github.com/rira2/DevifyX-/blob/main/Assets/loss%20after%20adversarial%20training%20.png)



## Bonus Features Implemented

As required by the assignment, the following bonus features were added:
- Hyperparameter tuning with Keras Tuner.
- Early stopping and model checkpointing to prevent overfitting.
- FGSM adversarial attack to test robustness.
- Simple adversarial training defense to improve performance on perturbed examples.


---

## How to Run

To run this notebook and reproduce the results:

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/digit-recognition-assignment.git
   cd digit-recognition-assignment
2.Install required packages:
pip install tensorflow keras-tuner seaborn

3.Open notebook.ipynb in Jupyter or Google Colab.

4.Run all cells step by step.

GPT Prompt Usage Summary
I used GPT (ChatGPT) mainly to get guidance on structuring the project, clarifying concepts like hyperparameter tuning and adversarial attacks, and for general troubleshooting help.
All final code, experiments, and results were run and checked by me.
