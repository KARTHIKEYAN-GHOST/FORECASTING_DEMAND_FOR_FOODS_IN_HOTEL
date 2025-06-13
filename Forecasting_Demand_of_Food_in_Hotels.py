import pandas as pd
import pickle as p
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('/content/large_hotel_food_demand.csv')

df.drop('Date',axis=1,inplace=True)

df.info()

label_encoders = {}
for col in ['Day', 'Weather', 'Event', 'Menu_Item']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

f_x = ['Day', 'Weather', 'Event', 'Holiday', 'Occupancy_Rate', 'Quantity_Sold']
X = df[f_x]
y = df['Menu_Item']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
target_names = [str(cls) for cls in label_encoders['Menu_Item'].classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

def pre_food():
    day = input("Enter day (e.g., Monday): ")
    weather = input("Enter weather (e.g., Stormy, Sunny, Cloudy, Rainy): ")
    event = input("Enter event (None, Music Night, Wedding, Conference, Festival): ")
    holiday = int(input("Is it a holiday? (1 for Yes, 0 for No): "))
    occupancy = int(input("Enter hotel occupancy rate (0-100): "))
    quantity = int(input("Enter quantity sold: "))

    day=day.capitalize()
    weather=weather.capitalize()
    event=event.capitalize()

    input_data = pd.DataFrame([[
        label_encoders['Day'].transform([day])[0],
        label_encoders['Weather'].transform([weather])[0],
        label_encoders['Event'].transform([event])[0],
        holiday,
        occupancy,
        quantity
    ]], columns=f_x)

    pred = model.predict(input_data)[0]
    demand_food = label_encoders['Menu_Item'].inverse_transform([pred])[0]
    return demand_food

p.dump(model,open('demand_food','wb'))
m=p.load(open('demand_food','rb'))
dem_food=pre_food()
print(f"\nDemand food: {dem_food}")
