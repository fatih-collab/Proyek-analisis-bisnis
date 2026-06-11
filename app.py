import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve

# ==========================================
# 1. PENGATURAN HALAMAN (Lebar & Judul)
# ==========================================
st.set_page_config(page_title="Telco Churn Dashboard", page_icon="📊", layout="wide")

st.title("📊 Telco Customer Churn Prediction")
st.write("Aplikasi ini dirancang untuk memprediksi dan menganalisis probabilitas pelanggan pindah/berhenti (Churn) menggunakan Machine Learning.")
st.markdown("---")

# ==========================================
# 2. MEMUAT DATA DAN MODEL (Gunakan Cache)
# ==========================================
@st.cache_data
def load_data():
    X_test = pd.read_csv('X_test_final.csv')
    y_test = pd.read_csv('y_test.csv')
    return X_test, y_test

@st.cache_resource
def load_models():
    log_model = joblib.load('log_model.pkl')
    dt_model = joblib.load('dt_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    return log_model, dt_model, rf_model

X_test, y_test = load_data()
log_model, dt_model, rf_model = load_models()

# ==========================================
# 3. MEMBUAT SIDEBAR UNTUK NAVIGASI
# ==========================================
st.sidebar.title("🧭 Navigasi Menu")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ("🏠 Dashboard Utama", "🎯 Evaluasi Model", "💡 Wawasan Bisnis")
)
st.sidebar.markdown("---")
st.sidebar.info("Gunakan menu di atas untuk berpindah halaman sesuai dengan informasi yang Anda butuhkan.")

# ==========================================
# 4. KONTEN BERDASARKAN PILIHAN SIDEBAR
# ==========================================

# ---------------------------------------------------------
# HALAMAN 1: DASHBOARD UTAMA
# ---------------------------------------------------------
if menu == "🏠 Dashboard Utama":
    st.header("🏠 Ringkasan Bisnis & Tren Pelanggan")
    
    # --- KPI CARDS ---
    total_pelanggan = len(y_test)
    jumlah_churn = int(y_test.iloc[:, 0].sum())
    persentase_churn = (jumlah_churn / total_pelanggan) * 100

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Pelanggan yang Dites", f"{total_pelanggan}", "Keseluruhan Data Test", delta_color="off")
    kpi2.metric("Jumlah Pelanggan Churn", f"{jumlah_churn}", "Data Aktual (Fakta)", delta_color="off")
    kpi3.metric("Persentase Churn", f"{persentase_churn:.1f}%", "Proporsi Pelanggan Churn", delta_color="off")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- GRAFIK TIPE KONTRAK & TENURE ---
    col_dash1, col_dash2 = st.columns(2)
    
    with col_dash1:
        st.subheader("Tingkat Churn Berdasarkan Tipe Kontrak")
        
        contract_cols = [col for col in X_test.columns if 'Contract' in col]
        if len(contract_cols) > 1:
            tipe_kontrak = X_test[contract_cols].idxmax(axis=1).str.replace('Contract_', '')
        elif 'Contract' in X_test.columns:
            tipe_kontrak = X_test['Contract']
        else:
            tipe_kontrak = ['Kontrak Tidak Ditemukan'] * len(X_test)
            
        df_contract = pd.DataFrame({'Tipe Kontrak': tipe_kontrak, 'Churn': y_test.iloc[:, 0].values})
        churn_rate = df_contract.groupby('Tipe Kontrak')['Churn'].mean().reset_index()
        churn_rate['Churn'] = churn_rate['Churn'] * 100
        
        fig_contract = px.bar(churn_rate, x='Tipe Kontrak', y='Churn', 
                              text_auto='.1f', color='Tipe Kontrak',
                              color_discrete_sequence=px.colors.qualitative.Pastel,
                              labels={'Churn': 'Persentase Churn (%)'})
        fig_contract.update_traces(textposition='outside', texttemplate='%{text}%')
        fig_contract.update_layout(height=450, showlegend=False, yaxis_range=[0, 100])
        st.plotly_chart(fig_contract, use_container_width=True)
        
    with col_dash2:
        st.subheader("Distribusi Lama Berlangganan (Tenure)")
        kolom_tenure = [col for col in X_test.columns if 'tenure' in col.lower()]
        
        if len(kolom_tenure) > 0:
            df_tenure = pd.DataFrame({
                'Tenure': X_test[kolom_tenure[0]],
                'Status': y_test.iloc[:, 0].map({1: 'Churn (Yes)', 0: 'Not Churn (No)'})
            })
            
            warna_cerah = {'Churn (Yes)': '#FF4B4B', 'Not Churn (No)': '#00CC96'} 
            fig_tenure = px.histogram(
                df_tenure, x='Tenure', color='Status', barmode='stack', nbins=30, 
                color_discrete_map=warna_cerah,
                labels={'Tenure': 'Lama Berlangganan (Bulan)', 'count': 'Jumlah Pelanggan'}
            )
            fig_tenure.update_layout(height=450, legend_title="Status Pelanggan",
                                     yaxis_title="Jumlah Pelanggan", xaxis_title="Tenure (Bulan)")
            st.plotly_chart(fig_tenure, use_container_width=True)


# ---------------------------------------------------------
# HALAMAN 2: EVALUASI MODEL
# ---------------------------------------------------------
elif menu == "🎯 Evaluasi Model":
    st.header("🎯 Evaluasi Performa Machine Learning")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kurva ROC (Perbandingan Model)")
        
        fig_roc = go.Figure()
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        
        models = {'Random Forest': rf_model, 'Logistic Regression': log_model, 'Decision Tree': dt_model}
        for name, model in models.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
            
        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=450)
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col2:
        st.subheader("Confusion Matrix")
        pilih_model = st.selectbox("Pilih Model untuk melihat detail tebakan:", 
                                   ("Random Forest", "Logistic Regression", "Decision Tree"))
        
        model_aktif = models[pilih_model]
        y_pred = model_aktif.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           title=f"Hasil Prediksi: {pilih_model}",
                           labels=dict(x="Tebakan Model", y="Kenyataan Asli"),
                           x=['Not Churn (0)', 'Churn (1)'], y=['Not Churn (0)', 'Churn (1)'])
        fig_cm.update_layout(height=450)
        st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{pilih_model}")
        
    st.markdown("---")
    
    # --- KESIMPULAN HASIL TRAINING & TESTING ---
    st.subheader("📝 Kesimpulan Evaluasi Performa Model")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info(
            "**📈 Analisis Kurva ROC**\n\n"
            "Berdasarkan visualisasi Kurva ROC, model **Random Forest** dan **Logistic Regression** menunjukkan performa yang paling dominan dan saling bersaing ketat. "
            "Garis kurva kedua model tersebut melengkung tajam mendekati sudut kiri atas, menandakan bahwa model memiliki tingkat kepekaan (*True Positive Rate*) yang tinggi. "
            "Ketiga model yang telah melalui proses *Hyperparameter Tuning* ini terbukti jauh melampaui garis tebakan acak (baseline), yang berarti model sangat handal dalam membedakan kelas pelanggan."
        )
        
    with col_info2:
        st.success(
            "**🧮 Analisis Confusion Matrix**\n\n"
            "Dari hasil Confusion Matrix, terlihat bahwa penerapan teknik penyeimbangan data (**SMOTE**) berhasil membuat model lebih sensitif dalam mendeteksi pelanggan Churn. "
            "Model **Random Forest** secara konsisten memberikan tingkat prediksi yang paling seimbang. Model ini mampu mendeteksi sebagian besar pelanggan yang benar-benar akan pergi (*True Positives*), "
            "sekaligus menekan jumlah tebakan salah / alarm palsu (*False Positives*) dengan lebih baik dibandingkan model Decision Tree."
        )


# ---------------------------------------------------------
# HALAMAN 3: WAWASAN BISNIS
# ---------------------------------------------------------
elif menu == "💡 Wawasan Bisnis":
    st.header("💡 Wawasan Bisnis & Kesimpulan")
    
    col_biz1, col_biz2 = st.columns([1.2, 1])
    
    with col_biz1:
        st.subheader("Variabel Paling Berpengaruh")
        st.write("Faktor-faktor yang paling memengaruhi keputusan pelanggan untuk Churn (berdasarkan model Random Forest).")
        
        try:
            importance = rf_model.best_estimator_.feature_importances_
        except AttributeError:
            importance = rf_model.feature_importances_
            
        df_imp = pd.DataFrame({'Feature': X_test.columns, 'Importance': importance})
        df_imp = df_imp.sort_values(by='Importance', ascending=False).head(10)
        
        fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                         color='Importance', color_continuous_scale='Magma')
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=450)
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_biz2:
        st.subheader("📝 Kesimpulan & Rekomendasi")
        
        st.info(
            "**1. Tipe Kontrak adalah Kunci**\n\n"
            "Pelanggan dengan sistem kontrak bulanan (*Month-to-month*) memiliki probabilitas *churn* yang jauh lebih tinggi dibandingkan kontrak 1 atau 2 tahun. Ini menunjukkan bahwa komitmen jangka pendek membuat pelanggan sangat mudah berpindah."
        )
        
        st.warning(
            "**2. Masa Kritis di Bulan-bulan Pertama**\n\n"
            "Mayoritas pelanggan yang memutuskan pergi (*churn*) terjadi pada saat *tenure* (lama berlangganan) masih sangat rendah. Jika pelanggan berhasil dipertahankan melewati masa kritis ini, probabilitas mereka untuk *churn* akan menurun drastis."
        )
        
        st.success(
            "**💡 Rekomendasi Tindakan Strategis:**\n"
            "* **Upselling Kontrak:** Tawarkan diskon atau *benefit* eksklusif bagi pelanggan yang bersedia beralih dari kontrak bulanan ke kontrak tahunan.\n"
            "* **Program Onboarding:** Buat program pelayanan ekstra (seperti pendampingan khusus atau *cashback*) pada 3-6 bulan pertama sejak pelanggan bergabung untuk meningkatkan rasa nyaman dan loyalitas mereka."
        )