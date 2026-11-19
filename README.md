# Hotel Food Demand Classification

This project uses a Random Forest classifier to predict a menu item label from hotel operating conditions and sales information. Despite the repository name using the word “forecasting”, the current script performs supervised classification of `Menu_Item`; it does not forecast a numeric quantity or a future time series.

## Dataset

`large_hotel_food_demand.csv` contains:

- `Date`
- `Day`
- `Menu_Item` — prediction target
- `Quantity_Sold`
- `Weather`
- `Event`
- `Holiday`
- `Occupancy_Rate`

The script drops `Date`, label-encodes `Day`, `Weather`, `Event`, and `Menu_Item`, and uses `Day`, `Weather`, `Event`, `Holiday`, `Occupancy_Rate`, and `Quantity_Sold` as input features.

## Workflow

1. Load the CSV and remove the `Date` column.
2. Encode the categorical columns with separate `LabelEncoder` instances.
3. Split the data into training and test sets (`test_size=0.2`, `random_state=42`).
4. Train a default `RandomForestClassifier`.
5. Print a classification report.
6. Save the fitted model to a file named `demand_food` using pickle.
7. Prompt for operating conditions and predict a menu item.

## Technology

- Python
- pandas
- scikit-learn
- pickle (Python standard library)

## Run locally

```bash
git clone https://github.com/KARTHIKEYAN-GHOST/FORECASTING_DEMAND_FOR_FOODS_IN_HOTEL.git
cd FORECASTING_DEMAND_FOR_FOODS_IN_HOTEL

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas scikit-learn
python Forecasting_Demand_of_Food_in_Hotels.py
```

## Prediction inputs

When prompted, enter:

- Day (for example, Monday)
- Weather (for example, Sunny or Rainy)
- Event (for example, None or Festival)
- Holiday flag (`1` or `0`)
- Occupancy rate (`0`–`100`)
- Quantity sold

The text values must exist in the encoder's training classes. An unseen day, weather condition, or event will cause the current encoder transformation to fail.

## Important limitations

- `Quantity_Sold` is an input feature. The model predicts the menu-item class, not the number of meals to prepare.
- The script saves and reloads the model, but the prediction function also depends on the label encoders held in memory. The encoders are not saved alongside the model.
- The current script does not persist a complete preprocessing pipeline. A production version should save the encoders and model together, validate inputs, and use a time-aware evaluation if the intended task is future demand prediction.
