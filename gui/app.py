import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("📊 ML Dashboard")

# =========================
# CHARGEMENT DES MODÈLES
# =========================
BASE_DIR = "../model"

sales_model = joblib.load(os.path.join(BASE_DIR, "sales_model.pkl"))
fraud_model = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
fraud_scaler = joblib.load(os.path.join(BASE_DIR, "fraud_scaler.pkl"))

menu = st.sidebar.radio("Menu", ["📈 Sales", "🚨 Fraud"])

# =========================================================
# SALES PREDICTION
# =========================================================
if menu == "📈 Sales":

    st.header("Prédiction des ventes")

    col1, col2, col3 = st.columns(3)

    with col1:
        quantity = st.number_input("Quantity", value=1.0)
        unit_price = st.number_input("UnitPrice", value=0.0)

    with col2:
        month = st.number_input("Month", value=1)
        day = st.number_input("Day", value=1)

    with col3:
        hour = st.number_input("Hour", value=0)
        weekday = st.number_input("WeekDay", value=1)

    features = np.array([[quantity, unit_price, month, day, hour, weekday]])

    if st.button("🚀 Predire la vente future"):

        try:
            pred = sales_model.predict(features)[0]

            st.success(f"📊 Vente prédite : {pred:.2f}")

            # 📊 Graph features
            fig, ax = plt.subplots()
            ax.bar(
                ["Qty", "Price", "Month", "Day", "Hour", "WeekDay"],
                features[0]
            )
            ax.set_title("Features Sales")
            st.pyplot(fig)

            # 📈 Prediction
            #fig2, ax2 = plt.subplots()
            #ax2.bar(["Prediction"], [pred])
            #ax2.set_title("Sales Prediction")
            #st.pyplot(fig2)

        except Exception as e:
            st.error(f"Erreur prédiction : {e}")


# =========================================================
# FRAUD DETECTION
# =========================================================
if menu == "🚨 Fraud":

    st.header("Détection de fraude")

    col1, col2, col3 = st.columns(3)

    with col1:
        quantity = st.number_input("Quantity", value=1.0, key="f1")

    with col2:
        unit_price = st.number_input("UnitPrice", value=0.0, key="f2")

    with col3:
        sales = st.number_input("Sales", value=0.0, key="f3")

    features = np.array([[quantity, unit_price, sales]])

    if st.button("🚨 Detecter fraude"):

        try:
            features = np.array([[quantity, unit_price, sales]])
            features_scaled = fraud_scaler.transform(features)

            pred = fraud_model.predict(features_scaled)[0]
            st.write("pred : ", pred)

            if pred == -1:
                st.error("🚨 FRAUDE DETECTÉE")
            else:
                st.success("✅ Transaction normale")

            # 📊 Graph
            fig, ax = plt.subplots()
            ax.bar(["Quantity", "UnitPrice", "Sales"], features[0])
            ax.set_title("Fraud Features")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur prédiction : {e}")