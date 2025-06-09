# Predicting Medical Appointment No-Shows using Bayesian Network Models

This project analyzes patient appointment records to predict whether a patient will **show up** or **miss** their scheduled medical appointment. The goal is to improve hospital planning and resource allocation by identifying potential no-shows in advance.

---

## 📊 Dataset

- **Name**: Medical Appointment No Shows Dataset  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments)  
- **Records**: ~110,000  
- **Context**: Appointments in Vitória, Brazil (2016)  
- **Target column**: `No-show` (Yes = did not show up, No = attended)

---

## 🧠 Models Used

The following Bayesian network models were built and compared:

- Naive Bayes
- Tree-Augmented Naive Bayes (TAN)
- Augmented Naive Bayes (ANB)
- PC Algorithm
- Bayesian Search

---

## ⚠️ Key Problem: Class Imbalance

Most patients **attended** their appointments, while very few **missed** them.  
This caused:

- High **accuracy** (~80%)  
- Very low **sensitivity** (~0%) for detecting no-shows  
- Very high **specificity** (>99%)  

➡️ The models failed to detect the minority class (no-shows), making them unreliable.

---

## 🛠️ Solution: SMOTE (Synthetic Minority Over-sampling Technique)

To balance the dataset, **SMOTE** was applied:

- Creates **synthetic no-show examples**
- Increases sensitivity
- Improves AUC (discrimination power)
- Slight trade-off in accuracy and specificity

---

## 🧪Evaluation

- 10-fold cross-validation was used
- Metrics reported: Accuracy, Sensitivity, Specificity, AUC
- Confusion matrices and ROC curves were analyzed
