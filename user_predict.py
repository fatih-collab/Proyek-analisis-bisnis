import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go

# =====================================================
# 1. SETUP PALET WARNA (Silakan ganti kode HEX di bawah ini)
# =====================================================
WARNA_TOMBOL_UTAMA = "#4F46E5"   # Ganti dengan warna tombol Submit/Predict
WARNA_TOMBOL_SIDEBAR = "#10B981" # Ganti dengan warna tombol Load Sample
WARNA_TEKS = "#1E3A8A"           # Ganti dengan warna teks angka metrik

# =====================================================
# 2. PENGATURAN HALAMAN & CUSTOM CSS
# =====================================================
st.set_page_config(
    page_title="ChurnGuard Analytics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memaksa CSS menggunakan !important agar tidak tertimpa tema bawaan
st.markdown(f"""
    <style>
    /* Styling Tombol Utama */
    div.stButton > button:first-child {{
        background-color: {WARNA_TOMBOL_UTAMA} !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        font-size: 18px !important;
        border: none !important;
    }}
    /* Styling Tombol Sidebar */
    [data-testid="stSidebar"] div.stButton > button:first-child {{
        background-color: {WARNA_TOMBOL_SIDEBAR} !important;
        color: white !important;
        font-size: 14px !important;
    }}
    .block-container {{
        padding-top: 2rem !important;
    }}
    </style>
""", unsafe_allow_html=True)


# =====================================================
# 3. INISIALISASI SESSION STATE
# =====================================================
if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False

def load_sample():
    st.session_state.sample_loaded = True

# =====================================================
# 4. SIDEBAR
# =====================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8633/8633215.png", width=80)
    st.title("💠 ChurnGuard")
    st.caption("Customer Retention Analytics Panel")
    st.markdown("---")
    
    st.subheader("🛠️ Quick Tools")
    st.write("Gunakan data sampel untuk melihat cara kerja model tanpa harus mengisi form manual.")
    st.button("⚡ Load Sample Data", on_click=load_sample, use_container_width=True)
    
    st.markdown("---")
    st.info("**Info Model:**\nAplikasi ini didukung oleh algoritma **Random Forest Classifier** yang telah dioptimasi.")

# =====================================================
# 5. LOAD MODEL & FEATURE COLUMNS
# =====================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("rf_model.pkl")
        try:
            model = model.best_estimator_
        except:
            pass
        return model
    except Exception as e:
        return None

rf_model = load_model()

FEATURE_COLUMNS = [
    'gender_Female','gender_Male', 'SeniorCitizen', 'Partner_No','Partner_Yes',
    'Dependents_No','Dependents_Yes', 'tenure', 'PhoneService_No','PhoneService_Yes',
    'MultipleLines_No','MultipleLines_No phone service','MultipleLines_Yes',
    'InternetService_DSL','InternetService_Fiber optic','InternetService_No',
    'OnlineSecurity_No','OnlineSecurity_No internet service','OnlineSecurity_Yes',
    'OnlineBackup_No','OnlineBackup_No internet service','OnlineBackup_Yes',
    'DeviceProtection_No','DeviceProtection_No internet service','DeviceProtection_Yes',
    'TechSupport_No','TechSupport_No internet service','TechSupport_Yes',
    'StreamingTV_No','StreamingTV_No internet service','StreamingTV_Yes',
    'StreamingMovies_No','StreamingMovies_No internet service','StreamingMovies_Yes',
    'Contract_Month-to-month','Contract_One year','Contract_Two year',
    'PaperlessBilling_No','PaperlessBilling_Yes', 'PaymentMethod_Bank transfer (automatic)',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'MonthlyCharges', 'TotalCharges'
]

# =====================================================
# 6. HEADER UTAMA
# =====================================================
st.title("🎯ChurnGuard - Live Customer Churn Prediction")
st.write("Pilih parameter pelanggan pada tab di bawah ini, lalu klik tombol **Predict Churn** untuk memproses data.")

if rf_model is None:
    st.error("🚨 Error: File `rf_model.pkl` tidak ditemukan! Pastikan file model berada di folder yang sama dengan script ini.")
    st.stop()

# =====================================================
# 7. FORM INPUT (TABS)
# =====================================================
val_gender = 0 if not st.session_state.sample_loaded else 1 
val_tenure = 12 if not st.session_state.sample_loaded else 2
val_contract = 0 if not st.session_state.sample_loaded else 0 
val_monthly = 70.0 if not st.session_state.sample_loaded else 95.5
val_total = 1000.0 if not st.session_state.sample_loaded else 191.0

tab1, tab2, tab3 = st.tabs(["👤 Profil Klien", "💳 Detail Pembayaran & Kontrak", "🌐 Paket Layanan"])

with tab1:
    st.markdown("### 📋 Data Demografi")
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=val_gender)
        senior = st.selectbox("Senior Citizen (Lansia)", [0, 1])
    with c2:
        partner = st.selectbox("Memiliki Pasangan (Partner)", ["Yes", "No"], index=1 if st.session_state.sample_loaded else 0)
        dependents = st.selectbox("Memiliki Tanggungan", ["Yes", "No"], index=1 if st.session_state.sample_loaded else 0)

with tab2:
    st.markdown("### 💸 Informasi Finansial")
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.number_input("Tenure (Lama Berlangganan dalam Bulan)", min_value=0, max_value=100, value=val_tenure)
        contract = st.selectbox("Tipe Kontrak", ["Month-to-month", "One year", "Two year"], index=val_contract)
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if st.session_state.sample_loaded else 0)
    with c2:
        payment = st.selectbox("Metode Pembayaran", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=val_monthly)
        total = st.number_input("Total Charges ($)", min_value=0.0, value=val_total)

with tab3:
    st.markdown("### 📶 Status Layanan Aktif")
    c1, c2, c3 = st.columns(3)
    with c1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with c2:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with c3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# 8. PEMETAAN INPUT DATA
# =====================================================
def create_input():
    data = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS)
    data["SeniorCitizen"] = senior
    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly
    data["TotalCharges"] = total
    
    try:
        data[f"gender_{gender}"] = 1
        data[f"Partner_{partner}"] = 1
        data[f"Dependents_{dependents}"] = 1
        data[f"PhoneService_{phone_service}"] = 1
        data[f"MultipleLines_{multiple_lines}"] = 1
        data[f"InternetService_{internet}"] = 1
        data[f"OnlineSecurity_{online_security}"] = 1
        data[f"OnlineBackup_{online_backup}"] = 1
        data[f"DeviceProtection_{device_protection}"] = 1
        data[f"TechSupport_{tech_support}"] = 1
        data[f"StreamingTV_{streaming_tv}"] = 1
        data[f"StreamingMovies_{streaming_movies}"] = 1
        data[f"Contract_{contract}"] = 1
        data[f"PaperlessBilling_{paperless}"] = 1
        data[f"PaymentMethod_{payment}"] = 1
    except KeyError as e:
        pass
    return data

# =====================================================
# 9. PREDIKSI & DASHBOARD HASIL
# =====================================================
if st.button("🚀 Process Prediction & Analyze", use_container_width=True):
    with st.spinner('Menganalisis pola pelanggan...'):
        try:
            input_df = create_input()
            
            # FITUR ANTI ERROR: Memastikan urutan kolom persis dengan model saat training
            if hasattr(rf_model, 'feature_names_in_'):
                input_df = input_df.reindex(columns=rf_model.feature_names_in_, fill_value=0)
            
            probability = rf_model.predict_proba(input_df)[0][1]
            prediction = rf_model.predict(input_df)[0]
            
            st.markdown("---")
            st.markdown("## 📊 Hasil Analisis Prediktif")
            
            col_chart, col_text = st.columns([1.2, 1.5])
            
            # VISUALISASI GAUGE CHART
            with col_chart:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    title = {'text': "Probabilitas Churn (%)", 'font': {'size': 20, 'color': WARNA_TEKS}},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "rgba(0,0,0,0)"}, 
                        'steps': [
                            {'range': [0, 30], 'color': "#10B981"},  
                            {'range': [30, 70], 'color': "#F59E0B"}, 
                            {'range': [70, 100], 'color': "#EF4444"} 
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': probability * 100
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            # TEXT & RECOMMENDATION
            with col_text:
                st.markdown("<br>", unsafe_allow_html=True)
                if probability < 0.30:
                    st.success(f"### ✅ LOW RISK ({probability:.1%})")
                    st.write("**Status:** Pelanggan ini sangat loyal. Tidak diperlukan tindakan intervensi khusus.")
                elif probability < 0.70:
                    st.warning(f"### ⚠️ MEDIUM RISK ({probability:.1%})")
                    st.write("**Status:** Pelanggan menunjukkan keraguan. Disarankan untuk menawarkan promosi kecil atau menanyakan kepuasan layanan via email/WA.")
                else:
                    st.error(f"### 🚨 HIGH RISK ({probability:.1%})")
                    st.write("**Status:** Pelanggan kemungkinan besar akan pindah/berhenti dalam waktu dekat. **Tindakan segera diperlukan!**")

            # LOGGING HISTORY
            history = pd.DataFrame({"Date": [datetime.now()], "Probability": [probability], "Prediction": [prediction]})
            if os.path.exists("prediction_history.csv"):
                history.to_csv("prediction_history.csv", mode="a", header=False, index=False)
            else:
                history.to_csv("prediction_history.csv", index=False)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")