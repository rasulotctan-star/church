import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Dataseti yükləyirik
df = pd.read_csv('customer_churn_dataset-testing-master.csv')

# ID sütununu silirik
df.drop('CustomerID', inplace=True, axis=1)

# Kateqoriyal sütunları rəqəmə çeviririk
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Subscription Type'] = df['Subscription Type'].map(
    {'Basic': 0, 'Standard': 1, 'Premium': 2})
df['Contract Length'] = df['Contract Length'].map(
    {'Monthly': 0, 'Annual': 1, 'Quarterly': 2})

# Data-nı bölürük
x = df.drop('Churn', axis=1)
y = df['Churn']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Modeli öyrədirik (Random Forest)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Ən əsas hissə: Modeli .pkl faylı olaraq yaddaşa yazırıq
with open('churn_rf_model.pkl', 'wb') as file:
    pickle.dump(rf, file)

print("✅ Model uğurla öyrədildi və 'churn_rf_model.pkl' olaraq yadda saxlanıldı!")