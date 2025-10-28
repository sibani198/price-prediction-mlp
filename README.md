# price-prediction-mlp
# 🏡 House Price Prediction using MLP and Other Models

This project was developed as part of **Assignment 1** for a **Kaggle-based competition** focused on predicting house prices using machine learning techniques.  
The objective was to analyze the dataset, build multiple models, tune their performance, and submit predictions for the hidden test set on Kaggle.

---

## 🎯 Project Overview

The task was to **predict the price of houses** based on various features provided in the dataset.  
The competition was hosted on **Kaggle**, and participants were required to:

- Analyze and preprocess the data  
- Handle missing values, duplicates, and outliers  
- Perform feature scaling and encoding  
- Train **at least 7 different models**  
- Perform **hyperparameter tuning** on **3 models**  
- Compare model performances and submit predictions  

---

## 📊 Approach

### 🧹 Data Exploration & Cleaning
- Identified data types of all columns  
- Checked and handled missing values using imputation techniques  
- Detected and removed duplicate entries  
- Identified outliers and handled them appropriately  
- Generated descriptive statistics for numerical features  

### 🧩 Feature Engineering
- Scaled numerical features using **standardization**  
- Encoded categorical features using **Label Encoding / One-Hot Encoding**  

### 🤖 Model Training
Trained **7 different machine learning models**, including:
1. Linear Regression  
2. Ridge Regression  
3. Lasso Regression  
4. Decision Tree Regressor  
5. Random Forest Regressor  
6. Gradient Boosting Regressor  
7. Multi-Layer Perceptron (MLP) Regressor  

### ⚙️ Hyperparameter Tuning
Performed tuning using **GridSearchCV** and **RandomizedSearchCV** on:
- Random Forest  
- Gradient Boosting  
- MLP  

### 📈 Model Evaluation
- Compared model performance based on **R² Score** and **Mean Squared Error (MSE)**  
- Final submission made using the **best-performing model (MLP)**  

---

## 🧠 Results

- Achieved an **R² score of 0.6443**, which was **remarkably close** to the target benchmark of **0.65** required for full marks.  
- Despite the narrow gap, the model demonstrated **strong predictive performance** and **generalization capabilities**.  
- The leaderboard score reflected **consistent improvement** across models after hyperparameter tuning.  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Platform:** Kaggle  

---

## 📈 Key Insights

- Feature scaling and encoding had a **significant impact** on model accuracy.  
- Ensemble methods like **Random Forest** and **Gradient Boosting** performed well, but the **MLP Regressor** slightly outperformed them after fine-tuning.  
- Handling outliers and missing data was crucial in improving performance.  

---

## 📹 Walkthrough

As per assignment requirements, a **video walkthrough (8–12 minutes)** was created, demonstrating:
- Data preprocessing steps  
- Visualizations and insights  
- Model training and tuning  
- Final evaluation and submission  
