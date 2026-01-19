# Hotel Food Demand Prediction using Random Forest

This project implements a machine learning model to predict **high-demand food menu items** in a large hotel based on operational and contextual factors such as day, weather, events, holidays, occupancy rate, and quantity sold.

The objective is to demonstrate an end-to-end ML workflow including preprocessing, model training, evaluation, serialization, and interactive prediction.

---

## Problem Statement

Hotels often struggle to anticipate which food items will be in high demand, leading to wastage or shortages.  
This project attempts to predict the most demanded **menu item** using historical hotel data and a Random Forest classifier.

---

## Dataset

- **File:** `large_hotel_food_demand.csv`
- **Target Column:** `Menu_Item`
- **Dropped Column:** `Date`

### Features Used
- Day  
- Weather  
- Event  
- Holiday  
- Occupancy_Rate  
- Quantity_Sold  

Categorical variables are encoded using **Label Encoding**.

---

## Workflow

1. Load and clean dataset
2. Encode categorical features
3. Split data into training and test sets
4. Train a Random Forest classifier
5. Evaluate model using classification metrics
6. Serialize trained model using Pickle
7. Accept user input and predict high-demand food item

---

## Model

- **Algorithm:** Random Forest Classifier
- **Library:** Scikit-learn
- **Train/Test Split:** 80/20
- **Evaluation Metric:** Precision, Recall, F1-score (via classification report)
### 1. Clone the repository
```bash
git clone <repositor
