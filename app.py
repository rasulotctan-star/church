from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Səhifənin başlığı və konfiqurasiyası
st.set_page_config(
    page_title="Müştəri İtkisi (Churn) Proqnozu", layout="centered")
st.title("📊 Müştəri İtkisi (Churn) Proqnozlaşdırma Sistemi")
st.write("Müştərinin məlumatlarını daxil edərək onun sistemdən ayrılma riskini yoxlayın.")

# 2. Öncədən öyrədilmiş Random Forest modelini yükləyirik


@st.cache_resource
def load_model():
    with open("churn_rf_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ 'churn_rf_model.pkl' faylı tapılmadı! Zəhmət olmasa əvvəlcə modeli öyrədib yaddaşa yazın.")
    st.stop()

# 3. İstifadəçi interfeysi - İnput sahələri
st.header("👤 Müştəri Məlumatları")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Yaş", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Cins", options=["Male", "Female"])
    tenure = st.number_input("İstifadə Müddəti (Ay)",
                             min_value=0, max_value=120, value=12, step=1)
    usage_frequency = st.number_input(
        "Aylıq İstifadə Tezliyi", min_value=0, max_value=30, value=15, step=1)
    support_calls = st.number_input(
        "Dəstək Zənglərinin Sayı", min_value=0, max_value=20, value=2, step=1)

with col2:
    payment_delay = st.number_input(
        "Ödəniş Gecikməsi (Gün)", min_value=0, max_value=30, value=0, step=1)
    subscription_type = st.selectbox("Abunəlik Növü", options=[
                                     "Basic", "Standard", "Premium"])
    contract_length = st.selectbox("Müqavilə Müddəti", options=[
                                   "Monthly", "Quarterly", "Annual"])
    total_spend = st.number_input(
        "Ümumi Xərc ($)", min_value=0.0, max_value=100000.0, value=500.0, step=50.0)
    last_interaction = st.number_input(
        "Son Aktivlikdən Keçən Gün", min_value=0, max_value=30, value=5, step=1)

# 4. Daxil edilən məlumatların modelin başa düşəcəyi formata çevrilməsi (Mapping)
gender_mapped = 0 if gender == "Male" else 1

sub_map = {"Basic": 0, "Standard": 1, "Premium": 2}
subscription_mapped = sub_map[subscription_type]

contract_map = {"Monthly": 0, "Annual": 1, "Quarterly": 2}
contract_mapped = contract_map[contract_length]

# Sütun sırası modelin öyrədildiyi x_train sırası ilə tamamilə eyni olmalıdır
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender_mapped,
    'Tenure': tenure,
    'Usage Frequency': usage_frequency,
    'Support Calls': support_calls,
    'Payment Delay': payment_delay,
    'Subscription Type': subscription_mapped,
    'Contract Length': contract_mapped,
    'Total Spend': total_spend,
    'Last Interaction': last_interaction
}])

st.markdown("---")

# 5. Proqnoz düyməsi (Prediction)
if st.button("🔮 Proqnozlaşdır", type="primary"):
    # Model vasitəsilə proqnoz alırıq
    prediction = model.predict(input_data)[0]

    # Ehtimal dərəcəsini yoxlayırıq (əgər model predict_proba dəstəkləyirsə)
    try:
        prediction_proba = model.predict_proba(input_data)[0][1] * 100
    except AttributeError:
        prediction_proba = None

    # Nəticənin ekrana çıxarılması
    if prediction == 1:
        st.error(
            f"⚠️ *Xəbərdarlıq:* Bu müştərinin sistemdən *AYRILMA (Churn)* ehtimalı yüksəkdir!")
        if prediction_proba:
            st.progress(prediction_proba / 100)
            st.write(f"Ayrılma riski: *{prediction_proba:.2f}%*")
    else:
        st.success(
            f"✅ *Müsbət:* Bu müştəri böyük ehtimalla sistemdə *QALACAQ (Loyal)*.")
        if prediction_proba:
            st.progress(prediction_proba / 100)
            st.write(
                f"Ayrılma riski çox aşağıdır: *{prediction_proba:.2f}%*")

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