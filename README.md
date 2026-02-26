🌿 Plant Disease Detection using Deep Learning
📌 Project Overview

This project implements a Convolutional Neural Network (CNN) model to detect and classify plant leaf diseases from images. The system helps in early identification of plant diseases, which can assist farmers and agricultural experts in taking preventive measures.

The model is trained on labeled leaf images and can predict disease categories from new test images.

🎯 Problem Statement

Plant diseases significantly reduce agricultural productivity. Manual detection is time-consuming and requires expert knowledge.

This project aims to build an automated image-based disease detection system using Deep Learning.

🧠 Model Architecture

Convolutional Neural Network (CNN)

Image preprocessing & normalization

Training & validation split

Accuracy and loss monitoring

Saved trained model for inference

🛠 Technologies Used

Python

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

Scikit-learn

📂 Project Structure
Plant-Disease-Detection/
│
├── train_plant_disease.ipynb
├── testing_plant_disease.ipynb
├── trained_model.keras
├── requirements.txt
└── README.md
🚀 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/your-username/Plant-Disease-Detection.git
cd Plant-Disease-Detection
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Train the model

Open:

train_plant_disease.ipynb

Run all cells to train the model.

4️⃣ Test the model

Open:

testing_plant_disease.ipynb

Upload a leaf image to predict the disease.

📊 Results

Model trained on labeled plant leaf dataset

Achieved high validation accuracy

Successfully classifies plant diseases from unseen images

(You can update this section with your exact accuracy)

🔍 Features

✔ Image preprocessing pipeline
✔ CNN-based classification
✔ Model saving and loading
✔ Prediction on custom images

📈 Future Improvements

Deploy as a Web Application

Convert into Mobile App

Use Transfer Learning (ResNet, EfficientNet)

Real-time disease detection using camera
